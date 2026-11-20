# Hospital Triage AI — Predição de Readmissão em 30 Dias

API que estima o **risco de um paciente voltar a ser internado em até 30 dias** após a alta e explica **quais fatores mais pesaram** nessa estimativa, para apoiar a equipe a priorizar o acompanhamento pós-alta.

![CI](https://github.com/danielinocencio536-png/hospital_triage_ai/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.12%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-009688)
![XGBoost](https://img.shields.io/badge/XGBoost-3.x-orange)
![SHAP](https://img.shields.io/badge/SHAP-explicabilidade-purple)
![AUC](https://img.shields.io/badge/AUC--ROC-0.79-brightgreen)

> **Aviso:** projeto de estudo com **dados sintéticos** inspirados na estrutura do MIMIC-IV. Não usar para decisões clínicas reais.

---

## Objetivo

Readmissões em 30 dias são caras e, muitas vezes, evitáveis. O objetivo do projeto é:

- **Identificar cedo** os pacientes com maior risco de readmissão, ainda na alta;
- **Explicar o porquê** de cada predição (SHAP), para que o profissional de saúde confie e possa agir;
- **Entregar via API**, pronta para ser integrada a um sistema hospitalar, com auditoria de cada predição.

## Minha contribuição

Projeto individual. Desenvolvi de ponta a ponta, em 4 etapas:

| Etapa | O que fiz |
|---|---|
| **Dia 1 — Dados e modelo** | Gerador de dados sintéticos com correlações clinicamente plausíveis (idade → comorbidades → tempo de internação → readmissão), prevalência calibrada em 15%. Treino com XGBoost, validação cruzada estratificada, escolha do limiar por Youden e cálculo de importância global com SHAP. |
| **Dia 2 — Infraestrutura** | Configuração por variáveis de ambiente (`pydantic-settings`), camada de banco assíncrona (PostgreSQL/SQLite) e contratos de entrada/saída com validação de faixas (Pydantic v2). IDs pseudonimizados pensando na LGPD. |
| **Dia 3 — API** | Serviço de inferência com explicação SHAP por paciente, rota `POST /api/v1/triage/predict`, `GET /health` para monitoramento e log de auditoria de cada predição. A API sobe em modo degradado se o modelo ou o banco falharem. |
| **Dia 4 — Testes e Docker** | Testes com pytest (pipeline, serviço e API, com o modelo treinado na própria suíte e SQLite no lugar do Postgres). Dockerfile multi-stage que treina o modelo durante o build e roda com usuário não-root, docker-compose com PostgreSQL e CI no GitHub Actions que sobe tudo e chama a API. |

**Decisões que tomei:**
- **XGBoost + SHAP** em vez de um modelo "caixa-preta" sem explicação: na saúde, a equipe precisa entender o motivo do alerta.
- **Limiar de Youden** em vez de 0,5 fixo: equilibra sensibilidade e especificidade numa base desbalanceada (15% de readmissões).
- **Arquitetura BCE** (Boundary-Control-Entity): separa API, regra de negócio e dados, o que facilita testes e manutenção.

## Evidências

### Métricas do modelo (conjunto de teste, 2.000 pacientes)
Fonte: [`models/metrics_report.json`](models/metrics_report.json)

| Métrica | Valor | O que significa |
|---|---|---|
| AUC-ROC | **0,79** | Boa capacidade de separar quem será e quem não será readmitido |
| Sensibilidade | **67,4%** | De cada 10 pacientes que voltam, o modelo sinaliza ~7 |
| Especificidade | 78,4% | Acerta a maioria dos que não voltam |
| VPN | **93,1%** | Quando o modelo diz "baixo risco", acerta 93% das vezes |
| Precisão (VPP) | 35,8% | Esperado com prevalência de 15%: o alerta serve para triagem, não para diagnóstico |

### Fatores que mais influenciam o risco (SHAP global)
Fonte: [`models/shap_global_importance.json`](models/shap_global_importance.json)

```
Idade                       ████████████████████████  0,575
Tipo de admissão            ████████████▌             0,300
Tempo de internação         ███████████▌              0,278
Nº de medicações            █████████▉                0,237
Insuficiência cardíaca      █████████▎                0,223
Doença renal crônica        ███████▋                  0,185
Diabetes                    ██████▍                   0,153
```

### Calibração: a probabilidade está superestimada
Fonte: [`models/evaluation_report.json`](models/evaluation_report.json), gerado por `python -m src.pipelines.evaluate_model` no mesmo conjunto de teste.

O modelo **ordena bem** os pacientes (AUC 0,79), mas a probabilidade que ele devolve **não deve ser lida ao pé da letra**. Na média ele prevê 36% de readmissão para uma base onde 15% voltam. O Brier (0,164) fica pior que o de um modelo que só repetisse a prevalência (0,129), e o erro de calibração (ECE) é 0,21.

```mermaid
xychart-beta
    title "Prevista x observada (10 grupos de 200 pacientes)"
    x-axis "Grupo, do menor para o maior risco previsto" [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    y-axis "Taxa de readmissão" 0 --> 0.9
    line [0.042, 0.100, 0.154, 0.209, 0.270, 0.337, 0.436, 0.544, 0.676, 0.845]
    line [0.020, 0.030, 0.060, 0.065, 0.085, 0.115, 0.100, 0.220, 0.310, 0.515]
```
Linha de cima: probabilidade média prevista. Linha de baixo: taxa real de readmissão.

A causa é o `scale_pos_weight`: ele dá peso ~5,6x aos readmitidos para melhorar a sensibilidade, mas isso infla todas as probabilidades. Num teste rápido, treinar sem o peso ou aplicar calibração isotônica derruba o Brier para ~0,105 e o ECE para ~0,02 sem perder AUC. Isso muda as faixas Low/Moderate/High (hoje 0,30 e 0,50), que foram pensadas na escala inflada, então ficou como próximo passo em vez de entrar escondido aqui.

### Fairness: erro por sexo e faixa etária
Mesmo limiar (0,50) para todos os grupos.

| Grupo | n | Readmissão real | AUC | Sensibilidade | Especificidade | VPP |
|---|---|---|---|---|---|---|
| Feminino | 933 | 14,9% | 0,78 | 64,0% | 81,4% | 37,6% |
| Masculino | 1.067 | 15,5% | 0,80 | 70,3% | 75,7% | 34,6% |
| 18-44 anos | 204 | 2,9% | 0,62 | 0% | 99,0% | — |
| 45-64 anos | 692 | 6,4% | 0,68 | 13,6% | 97,5% | 27,3% |
| 65-79 anos | 717 | 18,8% | 0,71 | 68,9% | 67,0% | 32,6% |
| 80+ anos | 387 | 30,7% | 0,77 | 89,1% | 41,4% | 40,3% |

- **Sexo:** diferença pequena (~6 pontos em sensibilidade e especificidade), coerente com a prevalência quase igual.
- **Idade:** aqui o modelo se comporta de forma bem diferente. Como a idade é o fator mais forte, um limiar único quase nunca alerta para quem tem menos de 65 anos (os 6 readmitidos de 18-44 passaram todos) e alerta demais acima de 80 (especificidade de 41%). O AUC dentro de cada faixa também é menor que o geral, ou seja, o modelo separa bem jovem de idoso, mas tem mais dificuldade de separar idosos entre si.
- Parte disso vem da base sintética, onde eu mesmo coloquei a idade como principal motor do risco. Com dados reais, o caminho seria olhar limiares por faixa ou features que diferenciem melhor o risco dos mais jovens. O grupo de 18-44 é pequeno (6 casos positivos), então esses números têm muita incerteza.

### Exemplo de resposta da API
```json
{
  "patient_id": "P-10482931",
  "risk_probability": 0.7124,
  "risk_category": "High",
  "readmission_predicted": true,
  "top_risk_factors": [
    { "feature": "comorbidity_heart_failure", "value": 1, "shap_value": 0.41, "direction": "increases_risk" }
  ],
  "model_version": "1.0.0"
}
```

## Como funciona uma predição

```mermaid
sequenceDiagram
    participant C as Cliente
    participant R as API (triage_router)
    participant S as TriageInferenceService
    participant M as XGBoost + SHAP
    participant DB as PostgreSQL

    C->>R: POST /api/v1/triage/predict (dados da alta)
    R->>R: valida o payload (Pydantic, 422 se inválido)
    R->>S: predict(paciente)
    S->>M: predict_proba + shap_values
    M-->>S: probabilidade e contribuições por coluna
    S->>S: soma o SHAP do one-hot, define Low/Moderate/High
    S-->>R: resultado + latência
    R->>DB: grava em triage_audit_logs
    alt banco fora do ar
        DB--xR: erro (a predição segue mesmo assim)
    end
    R-->>C: 200 + JSON + header X-Audit-Logged
```

## Estrutura

```
src/
├── pipelines/
│   ├── generate_mimic_data.py   # gera a base sintética
│   ├── train_model.py           # treina, avalia e salva o modelo + SHAP
│   └── evaluate_model.py        # calibração e fairness no conjunto de teste
├── core/config.py               # configurações (variáveis TRIAGE_*)
├── database/                    # conexão assíncrona e tabela de auditoria
├── models/triage_schemas.py     # contratos da API (Pydantic)
├── services/inference_service.py# predição + explicação SHAP
├── api/v1/triage_router.py      # rotas /triage/predict e /health
└── main.py                      # aplicação FastAPI
tests/                           # pytest: pipeline, serviço e API
docker/Dockerfile                # build multi-stage (treina o modelo no build)
docker-compose.yml               # API + PostgreSQL
models/                          # métricas, SHAP, calibração e fairness
```

## Como rodar

### Com Docker (API + PostgreSQL)

```bash
cp .env.example .env            # defina POSTGRES_PASSWORD
docker compose up --build -d
curl http://localhost:8000/health
```

O modelo é treinado dentro do build, então não precisa rodar nada antes. Em `production` o `/docs` fica desligado; para usar o Swagger, coloque `TRIAGE_ENVIRONMENT=development` no `.env`.

### Local

```bash
python -m venv .venv && .venv\Scripts\activate      # Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
copy .env.example .env                              # Linux/Mac: cp; use SQLite se não tiver PostgreSQL

python -m src.pipelines.generate_mimic_data --n-records 10000 --seed 42
python -m src.pipelines.train_model --data data/raw/mimic_synthetic.csv
python -m src.pipelines.evaluate_model                # calibração e fairness
uvicorn src.main:app --reload
```

Depois, abra **http://localhost:8000/docs** para testar a API pelo navegador.

### Testando com curl

```bash
curl -X POST http://localhost:8000/api/v1/triage/predict   -H "Content-Type: application/json"   -d '{
    "patient_id": "P-10482931",
    "age": 78,
    "gender": "F",
    "admission_type": "EMERGENCY",
    "length_of_stay": 9.5,
    "num_lab_procedures": 62,
    "num_medications": 21,
    "comorbidities": {"hypertension": true, "diabetes": true, "heart_failure": true, "coronary_artery_disease": true}
  }'
```

Comorbidades não informadas contam como `false`. Campos desconhecidos, idade abaixo de 18 ou `patient_id` com espaço/pontuação retornam 422.

### Testes

```bash
pip install -r requirements-dev.txt
python -m pytest
```

## Próximos passos

- [x] Testes automatizados (pytest) e CI no GitHub Actions
- [x] Dockerfile e docker-compose com PostgreSQL
- [x] Análise de calibração e de fairness por sexo e idade
- [ ] Calibrar as probabilidades (sem `scale_pos_weight` ou isotônica) e revisar as faixas de risco
- [ ] Painel simples para a equipe visualizar os pacientes de alto risco

## Autor

**Daniel Inocêncio** — [GitHub](https://github.com/danielinocencio536-png)
