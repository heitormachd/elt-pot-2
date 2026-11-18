from math import acos, asin, cos, pi, sin, sqrt

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad


frequencia_hz = 60.0
inercia_s = 5.0
passo_s = 0.0001
tempo_final_s = 5.0
potencia_mecanica = 1.0
tensao_barra_1 = 1.0
tensao_barra_2 = 1.0
reatancia_gerador = 0.20
reatancia_transformador = 0.10
reatancia_linha_1 = 0.40
reatancia_linha_2 = 0.40

velocidade_sincrona = 2 * pi * frequencia_hz
coeficiente_inercia = 2 * inercia_s / velocidade_sincrona
reatancia_gerador_trafo = reatancia_gerador + reatancia_transformador
reatancia_linhas = 1 / (1 / reatancia_linha_1 + 1 / reatancia_linha_2)

produto_tensoes = tensao_barra_1 * tensao_barra_2
angulo_barra_1 = asin(potencia_mecanica * reatancia_linhas / produto_tensoes)
fasor_barra_1 = tensao_barra_1 * complex(cos(angulo_barra_1), sin(angulo_barra_1))
corrente_pre_falta = (fasor_barra_1 - tensao_barra_2) / (1j * reatancia_linhas)
tensao_interna = fasor_barra_1 + 1j * reatancia_gerador_trafo * corrente_pre_falta
angulo_inicial = float(np.angle(tensao_interna))

reatancia_pre_falta = reatancia_gerador_trafo + reatancia_linhas
reatancia_pos_falta = reatancia_gerador_trafo + reatancia_linha_1
meia_linha_2 = reatancia_linha_2 / 2
acrescimo_reatancia_falta = reatancia_gerador_trafo * reatancia_linha_1 / meia_linha_2
reatancia_falta_meio = reatancia_pos_falta + acrescimo_reatancia_falta

produto_tensoes_internas = abs(tensao_interna) * tensao_barra_2
potencia_maxima_pre = produto_tensoes_internas / reatancia_pre_falta
potencia_maxima_pos = produto_tensoes_internas / reatancia_pos_falta
potencia_maxima_falta_meio = produto_tensoes_internas / reatancia_falta_meio
potencia_maxima_falta_barra = 0.0

angulo_equilibrio_pos = asin(potencia_mecanica / potencia_maxima_pos)
angulo_limite = pi - angulo_equilibrio_pos


def simular_euler(potencia_maxima_falta, amortecimento, tempo_abertura_s, duracao_s):
    numero_passos = round(duracao_s / passo_s)
    passo_abertura = round(tempo_abertura_s / passo_s)
    tempo_s = np.arange(numero_passos + 1) * passo_s
    angulo = np.zeros(numero_passos + 1)
    desvio_velocidade = np.zeros(numero_passos + 1)
    angulo[0] = angulo_inicial

    for indice in range(numero_passos):
        if indice < passo_abertura:
            potencia_maxima = potencia_maxima_falta
        else:
            potencia_maxima = potencia_maxima_pos

        potencia_eletrica = potencia_maxima * sin(angulo[indice])
        potencia_amortecimento = amortecimento * desvio_velocidade[indice]
        potencia_acelerante = potencia_mecanica - potencia_eletrica - potencia_amortecimento
        aceleracao = potencia_acelerante / coeficiente_inercia

        angulo[indice + 1] = angulo[indice] + passo_s * desvio_velocidade[indice]
        desvio_velocidade[indice + 1] = desvio_velocidade[indice] + passo_s * aceleracao

    return tempo_s, angulo, desvio_velocidade


def perde_sincronismo(angulo, desvio_velocidade, passo_abertura):
    angulo_pos = angulo[passo_abertura:]
    velocidade_pos = desvio_velocidade[passo_abertura:]
    cruzamentos = np.flatnonzero(angulo_pos >= angulo_limite)
    retornos = np.flatnonzero(velocidade_pos < 0)

    if cruzamentos.size and (not retornos.size or cruzamentos[0] < retornos[0]):
        return True
    if retornos.size:
        return False
    raise RuntimeError("A duração deve incluir a primeira oscilação pós-falta.")


def buscar_tempo_critico(potencia_maxima_falta, amortecimento):
    passo_estavel = 0
    passo_instavel = round(1.0 / passo_s)

    while passo_instavel - passo_estavel > 1:
        passo_teste = (passo_estavel + passo_instavel) // 2
        abertura_teste_s = passo_teste * passo_s
        _, angulo, desvio_velocidade = simular_euler(
            potencia_maxima_falta, amortecimento, abertura_teste_s, tempo_final_s
        )

        if perde_sincronismo(angulo, desvio_velocidade, passo_teste):
            passo_instavel = passo_teste
        else:
            passo_estavel = passo_teste

    return passo_estavel * passo_s, passo_instavel * passo_s


def angulo_critico_areas(potencia_maxima_falta):
    termo_mecanico = potencia_mecanica * (angulo_limite - angulo_inicial)
    termo_eletrico = potencia_maxima_pos * cos(angulo_limite)
    termo_eletrico -= potencia_maxima_falta * cos(angulo_inicial)
    diferenca_potencias = potencia_maxima_pos - potencia_maxima_falta
    return acos((termo_mecanico + termo_eletrico) / diferenca_potencias)


def integrando_tempo_em_falta(raiz_desvio_angulo):
    desvio_angulo = raiz_desvio_angulo**2
    angulo = angulo_inicial + desvio_angulo
    area_acelerante = potencia_mecanica * desvio_angulo
    area_acelerante += potencia_maxima_falta_meio * (cos(angulo) - cos(angulo_inicial))
    velocidade = sqrt(2 * area_acelerante / coeficiente_inercia)
    return 2 * raiz_desvio_angulo / velocidade


angulo_critico_meio = angulo_critico_areas(potencia_maxima_falta_meio)
angulo_critico_barra = angulo_critico_areas(potencia_maxima_falta_barra)
limite_integral = sqrt(angulo_critico_meio - angulo_inicial)
tempo_areas_meio_s, _ = quad(integrando_tempo_em_falta, 0, limite_integral)
aceleracao_falta_barra = potencia_mecanica / coeficiente_inercia
desvio_angulo_critico_barra = angulo_critico_barra - angulo_inicial
tempo_areas_barra_s = sqrt(2 * desvio_angulo_critico_barra / aceleracao_falta_barra)

print(f"Ângulo da barra 1 = {angulo_barra_1:.6f} rad")
print(f"Módulo da tensão interna = {abs(tensao_interna):.6f} pu")
print(f"Ângulo inicial = {angulo_inicial:.6f} rad")
print(f"Ângulo limite pós-falta = {angulo_limite:.6f} rad")
print(f"Tempo crítico pelas áreas iguais, falta no meio da linha = {tempo_areas_meio_s:.6f} s")
print(f"Tempo crítico pelas áreas iguais, falta na barra 1 = {tempo_areas_barra_s:.6f} s")

tempo_estavel_s, tempo_instavel_s = buscar_tempo_critico(potencia_maxima_falta_meio, 0.0)
print(f"Falta no meio da linha com D = 0 — {tempo_estavel_s:.4f} a {tempo_instavel_s:.4f} s")

for numero_simulacao, tempo_abertura_s in ((1, 0.32), (2, 0.33)):
    tempo_s, angulo, desvio_velocidade = simular_euler(
        potencia_maxima_falta_meio, 0.0, tempo_abertura_s, 3.0
    )
    desvio_angulo = angulo - angulo_inicial
    passo_abertura = round(tempo_abertura_s / passo_s)
    instavel = perde_sincronismo(angulo, desvio_velocidade, passo_abertura)
    resultado = "instável" if instavel else "estável"
    print(f"Simulação {numero_simulacao} — abertura em {tempo_abertura_s:.2f} s — {resultado}")

    figura, eixos = plt.subplots(3, 1, figsize=(7, 9), layout="constrained")
    titulo = f"Simulação {numero_simulacao} — abertura em {tempo_abertura_s:.2f} s"
    figura.suptitle(f"{titulo} — {resultado}")
    eixos[0].plot(tempo_s, desvio_angulo)
    eixos[0].set_ylabel("Δδ (rad)")
    eixo_angulo = eixos[0].secondary_yaxis(
        "right",
        functions=(lambda desvio: desvio + angulo_inicial, lambda angulo: angulo - angulo_inicial),
    )
    eixo_angulo.set_ylabel("δ (rad)")
    eixos[1].plot(tempo_s, desvio_velocidade)
    eixos[1].set_ylabel("Δω (rad/s)")
    for eixo in eixos[:2]:
        eixo.axvline(tempo_abertura_s, color="0.45", linestyle="--")
        eixo.set_xlabel("Tempo (s)")
        eixo.grid(True)

    eixos[2].plot(desvio_angulo, desvio_velocidade)
    eixos[2].set_xlabel("Δδ (rad)")
    eixos[2].set_ylabel("Δω (rad/s)")
    eixos[2].set_title("Retrato de fase")
    eixos[2].grid(True)

casos_criticos = [
    (3, 0.10, potencia_maxima_falta_meio, "falta no meio da linha 2"),
    (4, 0.05, potencia_maxima_falta_meio, "falta no meio da linha 2"),
    (5, 0.00, potencia_maxima_falta_barra, "falta na barra 1"),
]

for numero_simulacao, amortecimento, potencia_maxima_falta, local_falta in casos_criticos:
    tempo_estavel_s, tempo_instavel_s = buscar_tempo_critico(potencia_maxima_falta, amortecimento)
    print(f"Simulação {numero_simulacao} — {tempo_estavel_s:.4f} a {tempo_instavel_s:.4f} s")

    figura, eixos = plt.subplots(2, 2, figsize=(7, 6), sharex=True, layout="constrained")
    figura.suptitle(f"Simulação {numero_simulacao} • {local_falta} • D = {amortecimento:g}")

    casos = [(tempo_estavel_s - 0.002, "Estável"), (tempo_instavel_s + 0.002, "Instável")]
    for coluna, (tempo_abertura_s, resultado) in enumerate(casos):
        tempo_s, angulo, desvio_velocidade = simular_euler(
            potencia_maxima_falta, amortecimento, tempo_abertura_s, tempo_final_s
        )
        desvio_angulo = angulo - angulo_inicial

        eixos[0, coluna].set_title(f"{resultado} • abertura em {tempo_abertura_s:.4f} s")
        eixos[0, coluna].plot(tempo_s, desvio_angulo)
        eixos[0, coluna].set_ylabel("Δδ (rad)")
        eixos[1, coluna].plot(tempo_s, desvio_velocidade)
        eixos[1, coluna].set_ylabel("Δω (rad/s)")
        eixos[1, coluna].set_xlabel("Tempo (s)")
        for eixo in eixos[:, coluna]:
            eixo.axvline(tempo_abertura_s, color="0.45", linestyle="--")
            eixo.grid(True)

angulos = np.linspace(0, pi, 400)
plt.figure(figsize=(7, 4), layout="constrained")
plt.plot(angulos, potencia_maxima_pre * np.sin(angulos), label="Pré-falta")
plt.plot(angulos, potencia_maxima_pos * np.sin(angulos), label="Pós-falta")
plt.plot(angulos, potencia_maxima_falta_meio * np.sin(angulos), label="Falta no meio da linha 2")
plt.axhline(potencia_maxima_falta_barra, label="Falta na barra 1", color="C3")
plt.axhline(potencia_mecanica, label="Potência mecânica", color="0.35", linestyle="--")
plt.xlabel("Ângulo do rotor δ (rad)")
plt.ylabel("Potência ativa (pu)")
plt.legend(loc="lower center", bbox_to_anchor=(0.5, 1), ncols=3)
plt.grid(True)
plt.show()
