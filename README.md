# Otimização Solar para o Nordeste Brasileiro (2025-2075)

## 🌞 Sobre o Projeto

Este projeto visa **solucionar o déficit energético** na região Nordeste do Brasil através da **alocação estratégica de usinas solares fotovoltaicas** até o ano de 2075. Combinando projeções de demanda, análise geoespacial e modelos de otimização matemática, o sistema identifica:

- 📈 **Cenários futuros** de consumo energético
- 📍 **Melhores localizações** para novas usinas
- ⚡ **Capacidade necessária** por década
- 💰 **Custos otimizados** de implantação

---

## 🔍 Contexto do Problema

O Nordeste brasileiro enfrenta três desafios energéticos críticos:

1. **Demanda crescente**: Expansão econômica e populacional aumentam o consumo em ~3% ao ano
2. **Dependência hídrica**: Vulnerabilidade a crises hídricas e mudanças climáticas
3. **Potencial subutilizado**: Apenas 12% do potencial solar é explorado atualmente

*Exemplo do déficit projetado:*
```python
2025: 1.200 MW  
2035: 3.800 MW  
2075: 18.400 MW

## 📁 Bases de Dados Necessárias

### Fontes Oficiais
| Arquivo | Fonte | Descrição | Variáveis Chave |
|---------|-------|-----------|-----------------|
| `consumo_mensal_epe.csv` | [EPE](https://www.epe.gov.br/pt) | Consumo histórico de energia por região | `data`, `regiao`, `consumo_mwh`, `classe` |
| `capacidade_instalada_ons.csv` | [ONS](http://www.ons.org.br/) | Capacidade operacional de usinas | `fonte`, `potencia_mw`, `data_entrada`, `subsistema` |
| `irradiacao_nordeste.csv` | [INPE](http://www.inpe.br/) | Potencial solar por município | `municipio`, `irradiacao_kwh/m²`, `latitude`, `longitude` |

### Dados Complementares
| Arquivo | Fonte | Uso no Projeto |
|---------|-------|----------------|
| `areas_disponiveis_ne.csv` | [ANEEL](https://www.aneel.gov.br/) | Identificação de terrenos aptos |
| `tarifas_distribuidoras.csv` | [ANEEL](https://www.aneel.gov.br/) | Cálculo de viabilidade econômica |
| `projecao_populacional.csv` | [IBGE](https://www.ibge.gov.br/) | Ajuste de cenários de demanda |

> **Formato recomendado**: Todos os arquivos devem usar `;` como delimitador e codificação UTF-8

## ⚙️ O que o Código Faz

### 1. Pré-processamento (`carregar_dados.py`)
- **Filtragem geográfica**: Isola dados da região Nordeste
- **Tratamento de missing**: Preenche lacunas com:
  - Média móvel (dados temporais)
  - Interpolação espacial (dados geográficos)
- **Normalização**: Padroniza unidades para MW/kWh

### 2. Modelagem Preditiva (`projecao.py`)
```python
# Exemplo de modelo SARIMA implementado
modelo = SARIMAX(serie_historica, 
                order=(1,1,1), 
                seasonal_order=(1,1,1,12))