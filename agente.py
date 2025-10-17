import os
import streamlit as st
from crewai import Agent, Task, Crew, Process, LLM

st.header("Agentes de IA para Estudo")
st.write("Digite o Tema de Estudo")

api_key = os.getenv("GROQ_API_KEY") or st.secrets.get("groq_api_key")
if not api_key:
    st.warning("API key não definida. Configure a variável de ambiente GROQ_API_KEY ou adicione 'groq_api_key' em .streamlit/secrets.toml")

tema = st.text_input("Tema", placeholder="Digite o tema de estudo")
objetivo = st.text_input("Objetivo", placeholder="Estudar um tema específico")
nivel = st.text_input("Nível de detalhes", placeholder="Básico, Intermediário, Avançado")
objetivo_detalhado = st.text_area("Objetivo detalhado", placeholder="Descreva o objetivo de aprendizado")

# Botão para executar a ação (apenas um)
executar = st.button("Executar")

if executar:
    if not api_key:
        st.error("Execução cancelada: API key ausente. Defina GROQ_API_KEY ou 'groq_api_key' em .streamlit/secrets.toml")
    elif not tema:
        st.error("Por favor, insira o tema de estudo.")
    else:
        llm = LLM(
            model="groq/llama-3.1-8b-instant",
            api_key=api_key,
            temperature=0.2
        )

        agente_resumo = Agent(
            role="Redator de resumos didáticos",
            goal=f"Escrever um resumo claro e didático sobre o {tema} para o público de nível {nivel}, alinhado ao objetivo de aprendizado: {objetivo_detalhado}",
            backstory="Você é um especialista em educação e redação didática, com anos de experiência em simplificar conceitos complexos para estudantes de diferentes níveis.",
            llm=llm
        )
        agente_perguntas = Agent(
            role="Gerador de perguntas de estudo",
            goal=f"Gerar perguntas relevantes e desafiadoras sobre o {tema} para o nível {nivel}, focando no objetivo: {objetivo_detalhado}",
            backstory="Você é um pedagogo experiente, especializado em criar perguntas que estimulam o pensamento crítico e avaliam o entendimento dos alunos.",
            llm=llm
        )
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
        tarefa_perguntas = Task(
            description=f"Gere 5 perguntas de estudo sobre {tema} adequadas ao nível {nivel}, focando em {objetivo_detalhado}",
            agent=agente_perguntas,
            expected_output="Uma lista de 5 perguntas com respostas sugeridas."
        )
        tarefa_avaliacao = Task(
            description=f"Baseado no resumo e nas perguntas, avalie o potencial de aprendizado sobre {tema} para o nível {nivel}",
            agent=agente_avaliacao,
            expected_output="Uma avaliação do conteúdo gerado e sugestões de melhoria."
        )

        crew = Crew(
            agents=[agente_resumo, agente_perguntas, agente_avaliacao],
            tasks=[tarefa_resumo, tarefa_perguntas, tarefa_avaliacao],
            process=Process.sequential
        )

        resultado = crew.kickoff()

        st.subheader("Resultado dos Agentes")
        st.write("**Resumo:**")
        st.write(resultado.tasks_output[0].raw if len(resultado.tasks_output) > 0 else "Erro ao gerar resumo")
        st.write("**Perguntas:**")
        st.write(resultado.tasks_output[1].raw if len(resultado.tasks_output) > 1 else "Erro ao gerar perguntas")
        st.write("**Avaliação:**")
        st.write(resultado.tasks_output[2].raw if len(resultado.tasks_output) > 2 else "Erro ao gerar avaliação")