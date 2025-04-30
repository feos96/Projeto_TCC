# -*- coding: utf-8 -*-
"""
TCC: Otimização Solar para Suprir o Déficit do Nordeste em 2075
Versão Completa com Visualizações Avançadas
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
import folium
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.forecasting.stl import STLForecast
from sklearn.ensemble import RandomForestRegressor
from pyomo.environ import *
from sklearn.preprocessing import MinMaxScaler
import matplotlib.patches as mpatches

# Configurações gerais
plt.style.use('ggplot')
sns.set_palette("viridis")
pd.set_option('display.precision', 2)
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.family'] = 'DejaVu Sans'

# ======================
# 1. CARREGAMENTO DE DADOS
# ======================
def carregar_dados_nordeste():
    """Carrega e filtra dados específicos para o Nordeste"""
    try:
        # Dados de consumo (EPE)
        df_consumo = pd.read_csv('dados/consumo_mensal_epe.csv', parse_dates=['data'])
        df_consumo = df_consumo[df_consumo['regiao'] == 'Nordeste'].copy()
        df_consumo['ano'] = df_consumo['data'].dt.year
        
        # Dados de capacidade instalada (ONS)
        df_geracao = pd.read_csv('dados/capacidade_instalada_ons.csv', parse_dates=['data_entrada_operacao'])
        df_geracao = df_geracao[df_geracao['regiao'] == 'Nordeste'].copy()
        df_geracao['ano_entrada'] = df_geracao['data_entrada_operacao'].dt.year
        
        # Dados de irradiação solar (INPE)
        df_solar = pd.read_csv('dados/irradiacao_nordeste.csv')
        
        # Dados de áreas disponíveis (ANEEL/IBGE)
        df_areas = pd.read_csv('dados/areas_disponiveis_ne.csv')
        
        return df_consumo, df_geracao, df_solar, df_areas
    
    except Exception as e:
        print(f"Erro ao carregar dados: {e}")
        return None, None, None, None

# ======================
# 2. PROJEÇÃO DE DEMANDA
# ======================
def projetar_demanda_2075(df_consumo):
    """Projeção robusta para 50 anos com múltiplas técnicas"""
    consumo_anual = df_consumo.groupby('ano')['consumo_mwh'].sum()
    
    # Modelagem
    modelo_sarima = SARIMAX(consumo_anual, order=(1,1,1), seasonal_order=(1,1,1,5)).fit(disp=False)
    modelo_stl = STLForecast(consumo_anual, RandomForestRegressor(), period=5).fit()
    
    # Projeções
    forecast_sarima = modelo_sarima.get_forecast(steps=50)
    forecast_stl = modelo_stl.get_forecast(steps=50)
    
    # Cenários
    cenarios = {
        'conservador': consumo_anual.iloc[-1] * (1.01 ** np.arange(1,51)),
        'moderado': consumo_anual.iloc[-1] * (1.03 ** np.arange(1,51)),
        'otimista': consumo_anual.iloc[-1] * (1.05 ** np.arange(1,51))
    }
    
    return {
        'sarima': forecast_sarima.predicted_mean,
        'stl': forecast_stl.predicted_mean,
        'cenarios': pd.DataFrame(cenarios),
        'intervalo_confianca': forecast_sarima.conf_int()
    }

# ======================
# 3. ANÁLISE DE POTENCIAL
# ======================
def analisar_potencial_solar(df_solar, df_areas):
    """Calcula o potencial solar teórico"""
    df = pd.merge(df_areas, df_solar, on='municipio')
    df['potencial_mw'] = df['irradiacao'] * df['area_km2'] * 0.18 * 365 / 1000
    return df.nlargest(50, 'potencial_mw')

# ======================
# 4. OTIMIZAÇÃO
# ======================
def otimizar_alocacao(demanda, df_potencial):
    """Modelo Pyomo para alocação ótima"""
    model = ConcreteModel()
    model.L = Set(initialize=df_potencial.index)
    model.T = Set(initialize=range(2025,2076))
    
    # Parâmetros
    model.demanda = Param(model.T, initialize=demanda)
    model.potencial = Param(model.L, initialize=df_potencial['potencial_mw'].to_dict())
    
    # Variáveis
    model.alocar = Var(model.L, model.T, domain=Binary)
    model.capacidade = Var(model.L, model.T, domain=NonNegativeReals)
    
    # Função objetivo
    def objetivo_rule(model):
        return sum(model.capacidade[l,t] for l in model.L for t in model.T)
    model.objetivo = Objective(rule=objetivo_rule, sense=maximize)
    
    # Restrições
    def demanda_rule(model, t):
        return sum(model.capacidade[l,t] for l in model.L) >= model.demanda[t] * 0.8
    model.rest_demanda = Constraint(model.T, rule=demanda_rule)
    
    # Resolver
    solver = SolverFactory('glpk')
    results = solver.solve(model)
    
    # Processar resultados
    alocacao = pd.DataFrame(index=model.T, columns=df_potencial['municipio'])
    for t in model.T:
        for l in model.L:
            alocacao.at[t, df_potencial.at[l,'municipio']] = model.capacidade[l,t]()
    
    return alocacao

# ======================
# 5. VISUALIZAÇÕES (NOVO)
# ======================
def gerar_visualizacoes(projecoes, alocacao, df_potencial):
    """Gera todos os gráficos e mapas"""
    # 1. Gráfico de Cascata
    plot_waterfall(projecoes, alocacao)
    
    # 2. Heatmap Temporal
    plot_heatmap_evolucao(alocacao)
    
    # 3. Gráfico de Radar
    plot_radar_potencial(df_potencial)
    
    # 4. Mapa Interativo
    plot_mapa_interativo(alocacao, df_potencial)
    
    # 5. Trajetória de Capacidade
    plot_trajetoria_capacidade(alocacao)

def plot_waterfall(projecoes, alocacao):
    """Gráfico de cascata do déficit energético"""
    demanda_2075 = projecoes['cenarios']['moderado'].iloc[-1]
    capacidade_2075 = alocacao.sum(axis=1).iloc[-1]
    
    fig = go.Figure(go.Waterfall(
        name="Déficit 2075",
        orientation="v",
        measure=["absolute", "relative", "relative", "total"],
        x=["Demanda Projetada", "Geração Atual", "Expansão Solar", "Déficit Residual"],
        textposition="outside",
        y=[demanda_2075, -projecoes['sarima'].iloc[0], capacidade_2075, -demanda_2075 + capacidade_2075],
        connector={"line":{"color":"rgb(63, 63, 63)"}},
    ))
    
    fig.update_layout(
        title="Resolução do Déficit Energético no Nordeste (2075)",
        yaxis_title="Energia (MWh)",
        showlegend=False
    )
    fig.write_image("resultados/waterfall.png")

def plot_heatmap_evolucao(alocacao):
    """Heatmap da expansão por década"""
    alocacao['decada'] = (alocacao.index // 10) * 10
    df_decadal = alocacao.groupby('decada').sum().T
    
    plt.figure(figsize=(12,8))
    sns.heatmap(df_decadal, cmap="YlOrBr", annot=True, fmt=".0f", linewidths=.5)
    plt.title("Expansão da Capacidade Solar por Município (2025-2075)")
    plt.xlabel("Década")
    plt.ylabel("Municípios Prioritários")
    plt.tight_layout()
    plt.savefig("resultados/heatmap.png", dpi=300)

def plot_radar_potencial(df_potencial):
    """Gráfico de radar comparativo"""
    cols = ['irradiacao', 'area_km2', 'distancia_subestacao_km']
    df_norm = df_potencial[cols].apply(lambda x: x/x.max())
    
    angles = np.linspace(0, 2*np.pi, len(cols), endpoint=False).tolist()
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(8,8), subplot_kw={'polar': True})
    
    for i in df_potencial.nlargest(3, 'potencial_mw').index:
        values = df_norm.loc[i].tolist()
        values += values[:1]
        ax.plot(angles, values, label=df_potencial.loc[i, 'municipio'])
        ax.fill(angles, values, alpha=0.25)
    
    ax.set_theta_offset(np.pi/2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), cols)
    ax.legend()
    plt.savefig("resultados/radar.png")

# ======================
# 6. FLUXO PRINCIPAL
# ======================
def main():
    print("Iniciando análise para o Nordeste (2075)...")
    
    # 1. Carregar dados
    df_consumo, df_geracao, df_solar, df_areas = carregar_dados_nordeste()
    
    # 2. Projetar demanda
    print("Projetando demanda...")
    projecoes = projetar_demanda_2075(df_consumo)
    
    # 3. Analisar potencial
    print("Analisando potencial solar...")
    df_potencial = analisar_potencial_solar(df_solar, df_areas)
    
    # 4. Otimizar alocação
    print("Otimizando alocação...")
    alocacao = otimizar_alocacao(projecoes['cenarios']['moderado'], df_potencial)
    
    # 5. Gerar visualizações
    print("Gerando visualizações...")
    gerar_visualizacoes(projecoes, alocacao, df_potencial)
    
    print("Análise concluída! Verifique a pasta 'resultados'.")

if __name__ == "__main__":
    main()