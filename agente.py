import os
import streamlit as st
from crewai import Agent, Task, Crew, Process, LLM

st.header("Agentes de IA para Estudo")
st.write("Digite o Tema de Estudo")

api_key = os.getenv('')

tema = st.text_input("Tema", placeholder="Digite o tema de estudo")

objetivo = st.text_input("Objetivo", placeholder="Estudar um tema específico")

nivel = st.text_input("Nível de detalhes", placeholder="Básico, Intermediário, Avançado")

objetivo_detalhado = st.text_area("Objetivo detalhado", placeholder="Descreva o objetivo de aprendizado")

# Botão para executar a ação
executar = st.button("Executar")

# Verifica se o botão foi pressionado
if executar:
    if not tema:
        st.error("Por favor, insira o tema de estudo.")
    else:
        llm = LLM(
            model="groq/llama-3.1-8b-instant",
            api_key=api_key,
            temperature=0.2  # Define o nível de criatividade das respostas
        )

        agente_resumo = Agent(
            role="Redator de resumos didáticos",
            goal=f"Escrever um resumo claro e didático sobre o {tema} para o público de nível {nivel}, alinhado ao objetivo de aprendizado: {objetivo_detalhado}",
            backstory="Você é um especialista em educação e redação didática, com anos de experiência em simplificar conceitos complexos para estudantes de diferentes níveis.",
            llm=llm
        )
        # Gerador de perguntas
        agente_perguntas = Agent(
            role="Gerador de perguntas de estudo",
            goal=f"Gerar perguntas relevantes e desafiadoras sobre o {tema} para o nível {nivel}, focando no objetivo: {objetivo_detalhado}",
            backstory="Você é um pedagogo experiente, especializado em criar perguntas que estimulam o pensamento crítico e avaliam o entendimento dos alunos.",
            llm=llm
        )
        # Avaliador de aprendizado
        agente_avaliacao = Agent(
            role="Avaliador de aprendizado",
            goal=f"Avaliar o progresso de aprendizado no {tema} com base no resumo e perguntas geradas, para o nível {nivel}",
            backstory="Você é um avaliador educacional com expertise em analisar conteúdos de aprendizado e fornecer feedback construtivo para melhoria.",
            llm=llm
        )

        tarefa_resumo = Task(
            description=f"Crie um resumo detalhado sobre {tema} no nível {nivel}, alinhado ao objetivo: {objetivo_detalhado}",
            agent=agente_resumo,
            expected_output="Um resumo claro e didático em formato de texto."
        )
        # Geração de perguntas
        tarefa_perguntas = Task(
            description=f"Gere 5 perguntas de estudo sobre {tema} adequadas ao nível {nivel}, focando em {objetivo_detalhado}",
            agent=agente_perguntas,
            expected_output="Uma lista de 5 perguntas com respostas sugeridas."
        )
        # Avaliação
        tarefa_avaliacao = Task(
            description=f"Baseado no resumo e nas perguntas, avalie o potencial de aprendizado sobre {tema} para o nível {nivel}",
            agent=agente_avaliacao,
            expected_output="Uma avaliação do conteúdo gerado e sugestões de melhoria."
        )
        # Configuração do Crew com os agentes e tarefas
        crew = Crew(
            agents=[agente_resumo, agente_perguntas, agente_avaliacao],
            tasks=[tarefa_resumo, tarefa_perguntas, tarefa_avaliacao],
            process=Process.sequential
        )
        # Execução do Crew
        resultado = crew.kickoff()

        st.subheader("Resultado dos Agentes")
        st.write("**Resumo:**")
        st.write(resultado.tasks_output[0].raw if len(resultado.tasks_output) > 0 else "Erro ao gerar resumo")
        st.write("**Perguntas:**")
        st.write(resultado.tasks_output[1].raw if len(resultado.tasks_output) > 1 else "Erro ao gerar perguntas")
        st.write("**Avaliação:**")
        st.write(resultado.tasks_output[2].raw if len(resultado.tasks_output) > 2 else "Erro ao gerar avaliação")