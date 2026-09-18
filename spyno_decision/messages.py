"""F5 — Geração do CONTEÚDO da notificação (o "caminho A").

Separação de papéis, seguindo Grundmann et al. (arXiv:2506.07275, cMABxLLM):

    o BANDIT decide QUANDO notificar e QUE TIPO de gatilho usar
    o LLM escreve O TEXTO daquele tipo, adaptado ao contexto

Por que esta separação importa (e por que é o uso defensável do LLM neste
trabalho): a decisão fica com um modelo estatístico auditável, com propensity
logada e avaliável por OPE; o LLM atua só onde é comprovadamente bom —
linguagem. É o oposto do papel em que ele falhou aqui (simular usuários:
ver `spyno-heartsteps/run_generator_arbitration.py` e arXiv:2606.20708).

Motivação empírica no próprio trabalho: o dado real mostra que o efeito da
notificação DECAI ao longo das semanas (habituação, β=−1,23). Conteúdo
repetitivo é uma causa plausível; variedade é uma intervenção plausível. Este
módulo permite medir a variedade (F5b) e, no piloto, testar a hipótese.

Dois geradores com a mesma interface:
  TemplateMessages — banco fixo de frases por tipo (baseline; é o que o
                     sistema faz hoje). Sem custo, sem variedade.
  LLMMessages      — Claude via OpenRouter, condicionado a tipo + contexto +
                     perfil, com cache em disco.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

from .policy import ARM_NAMES

# --- definição dos tipos (taxonomia de gatilhos de Fogg, 2009) --------------
ARM_BRIEF = {
    "spark": {
        "objetivo": "ELEVAR A MOTIVAÇÃO de quem consegue treinar mas está sem vontade",
        "faz": "lembra do porquê, do progresso já feito, do que a pessoa ganha",
        "nao_faz": "não sugere encurtar o treino nem facilitar (a habilidade não é o problema)",
    },
    "facilitator": {
        "objetivo": "REDUZIR O ESFORÇO de quem quer treinar mas está sem tempo/energia",
        "faz": "oferece uma versão menor e concreta: treino curto, só um exercício, em casa",
        "nao_faz": "não apela para motivação nem cobra (a vontade não é o problema)",
    },
    "signal": {
        "objetivo": "APENAS LEMBRAR quem já está motivado e disponível",
        "faz": "avisa que é a hora, de forma curta e neutra",
        "nao_faz": "não persuade, não explica, não oferece alternativa",
    },
}

SYSTEM = """Você escreve notificações push de um app de treino brasileiro.

REGRAS INEGOCIÁVEIS:
- Máximo 90 caracteres. É uma notificação, não uma mensagem.
- Português do Brasil, tom de colega — nunca de coach motivacional exagerado.
- Nada de culpa, cobrança, alarme ou promessa de resultado ("você vai secar").
- No máximo 1 emoji, e só se couber naturalmente. Prefira nenhum.
- Não invente dados que você não recebeu (peso, medidas, metas).
- Escreva APENAS o texto da notificação, sem aspas e sem explicação.

Você receberá o TIPO de gatilho a usar e o contexto da pessoa. O tipo é uma
decisão já tomada por outro sistema: respeite-o à risca."""

USER_TEMPLATE = """TIPO: {arm}
Objetivo do tipo: {objetivo}
O que faz: {faz}
O que NÃO faz: {nao_faz}

CONTEXTO AGORA:
{contexto}

Escreva a notificação."""


def _describe(ctx: dict) -> str:
    """Contexto legível — só sinais que o app realmente coleta."""
    linhas = []
    h = int(ctx.get("hour", 12))
    periodo = ("de manhã" if 6 <= h < 12 else "à tarde" if 12 <= h < 18
               else "à noite" if 18 <= h < 22 else "de madrugada")
    linhas.append(f"- horário: {h}h ({periodo}){', fim de semana' if ctx.get('is_weekend') else ''}")
    if ctx.get("in_free_window"):
        linhas.append("- está numa janela livre da rotina dela")
    if ctx.get("location_home"):
        linhas.append("- está em casa")
    dias = ctx.get("days_since_last")
    if dias is not None:
        linhas.append(f"- último treino: {'hoje' if dias == 0 else f'há {dias} dia(s)'}")
    streak = int(ctx.get("streak", 0))
    if streak >= 2:
        linhas.append(f"- está numa sequência de {streak} dias treinando")
    elif streak == 0 and (dias or 0) >= 3:
        linhas.append("- perdeu a sequência que tinha")
    if ctx.get("barreira"):
        linhas.append(f"- barreira que ela relata: {ctx['barreira']}")
    if ctx.get("notifs_today", 0) >= 2:
        linhas.append("- já recebeu outras notificações hoje (seja mais leve)")
    return "\n".join(linhas)


class TemplateMessages:
    """Baseline: banco fixo de frases por tipo (o que o sistema faz hoje)."""

    name = "template"

    BANK = {
        "spark": ["Bora treinar? Seu último treino foi ótimo 💪",
                  "Hora de manter o ritmo!"],
        "facilitator": ["Treino curto de 15 min pronto pra agora",
                        "Que tal só o aquecimento hoje?"],
        "signal": ["Hora do seu treino", "Seu treino está te esperando"],
    }

    def __init__(self, seed: int = 0):
        self._i = 0
        self.calls = 0

    def write(self, arm: int, ctx: dict) -> str:
        name = ARM_NAMES[arm]
        bank = self.BANK[name]
        msg = bank[self._i % len(bank)]
        self._i += 1
        self.calls += 1
        return msg


class LLMMessages:
    """Claude via OpenRouter escreve a mensagem do tipo escolhido pelo bandit."""

    name = "llm"

    def __init__(self, model: str = "anthropic/claude-opus-4.8",
                 cache_dir: str = ".msg_cache"):
        from openrouter import OpenRouter  # lazy

        self._client = OpenRouter(api_key=os.environ["OPENROUTER_API_KEY"])
        self._model = model
        self._cache = Path(cache_dir)
        self._cache.mkdir(exist_ok=True)
        self.calls = 0
        self.tokens_in = 0
        self.tokens_out = 0

    def write(self, arm: int, ctx: dict) -> str:
        name = ARM_NAMES[arm]
        if name == "no_notify":
            return ""
        brief = ARM_BRIEF[name]
        user = USER_TEMPLATE.format(arm=name, contexto=_describe(ctx), **brief)
        key = hashlib.sha256(f"{self._model}|{user}".encode()).hexdigest()[:24]
        cached = self._cache / f"{key}.txt"
        if cached.exists():
            return cached.read_text(encoding="utf-8")

        last = None
        for attempt in range(4):
            try:
                resp = self._client.chat.send(
                    model=self._model, max_tokens=200,
                    messages=[{"role": "system", "content": SYSTEM},
                              {"role": "user", "content": user}],
                )
                text = resp.choices[0].message.content.strip().strip('"')
                self.calls += 1
                self.tokens_in += resp.usage.prompt_tokens
                self.tokens_out += resp.usage.completion_tokens
                cached.write_text(text, encoding="utf-8")
                return text
            except Exception as e:
                last = e
                time.sleep(2 ** attempt)
        raise RuntimeError(f"OpenRouter falhou após 4 tentativas: {last}")


def make_writer(mode: str, cache_dir: str = ".msg_cache"):
    """mode: 'template' | 'llm'."""
    return TemplateMessages() if mode == "template" else LLMMessages(cache_dir=cache_dir)
