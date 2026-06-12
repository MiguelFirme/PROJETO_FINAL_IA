"""
Agente Híbrido com Gemini Function Calling (arquitetura MCP-like)
Ferramentas:
  - buscar_filmes_rag : busca semântica no FAISS local
  - buscar_na_web     : busca na internet via DuckDuckGo
O Gemini decide autonomamente qual ferramenta chamar — sem if/else manual.
"""
import os
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# ── Configuração ──────────────────────────────────────────────────────────────

MINHA_CHAVE_GEMINI = "AIzaSyDMyxiP63AAH0rKH8J9GXrx5cOgxydByJI"
genai.configure(api_key=MINHA_CHAVE_GEMINI, transport='rest')


# ── Ferramentas reais (executadas pelo Python) ────────────────────────────────

def buscar_filmes_rag(query: str, k: int = 3) -> str:
    """Busca filmes no banco FAISS local por similaridade semântica."""
    try:
        db_path = "FAISS_DB"
        if not os.path.exists(db_path):
            return "Erro: pasta FAISS_DB não encontrada. Execute Alimentacao.py primeiro."

        embeddings = HuggingFaceEmbeddings(model_name="paraphrase-multilingual-MiniLM-L12-v2")
        
        db = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)
        docs = db.similarity_search(query, k=k)
        if not docs:
            return "Nenhum filme encontrado para essa busca."
        return "\n".join(f"🎥 {d.page_content}" for d in docs)
    except Exception as e:
        return f"Erro no RAG: {e}"


def buscar_na_web(query: str) -> str:
    """Busca informações atuais na internet via DuckDuckGo."""
    try:
        from ddgs import DDGS
        with DDGS() as ddgs:
            resultados = list(ddgs.text(query, max_results=4))
        if not resultados:
            return "Nenhum resultado encontrado na web."
        return "\n\n".join(
            f"📰 {r['title']}\n{r['body']}\nFonte: {r['href']}"
            for r in resultados
        )
    except Exception as e:
        return f"Erro na busca web: {e}"


# ── Mapa de execução: nome → função Python ───────────────────────────────────

EXECUTORES = {
    "buscar_filmes_rag": buscar_filmes_rag,
    "buscar_na_web":     buscar_na_web,
}


# ── Declaração das ferramentas para o Gemini (Function Calling) ───────────────

TOOLS_GEMINI = [
    {
        "function_declarations": [
            {
                "name": "buscar_filmes_rag",
                "description": (
                    "Busca filmes no banco de dados local. "
                    "Use para recomendações por gênero, humor, tema ou sentimento. "
                    "Ex: 'filmes de terror', 'comédia romântica', 'aventura espacial'."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Texto de busca para encontrar filmes."
                        },
                        "k": {
                            "type": "integer",
                            "description": "Número de resultados desejados (padrão: 3)."
                        }
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "buscar_na_web",
                "description": (
                    "Busca informações atuais na internet. "
                    "Use para perguntas sobre diretores, elenco, sinopses, prêmios, "
                    "lançamentos recentes ou qualquer informação não disponível localmente."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Termo de busca. Ex: 'diretor de Oppenheimer'."
                        }
                    },
                    "required": ["query"]
                }
            }
        ]
    }
]

SYSTEM_PROMPT = """Você é um assistente especialista em filmes.
Você tem acesso a duas ferramentas:
- buscar_filmes_rag: para recomendações e buscas no catálogo local
- buscar_na_web: para informações sobre diretores, elenco, prêmios, sinopses etc.
Use as ferramentas sempre que precisar. Responda em português, de forma objetiva e simpática."""


# ── Agentic loop ──────────────────────────────────────────────────────────────

def executar_ferramenta(nome: str, args: dict) -> str:
    """Executa a função Python correspondente ao nome da ferramenta."""
    func = EXECUTORES.get(nome)
    if not func:
        return f"Ferramenta desconhecida: {nome}"
    return func(**args)


def main():
    print("🤖 Iniciando Agente Gemini Function Calling (MCP-like)...")

    modelo = genai.GenerativeModel(
        model_name="models/gemini-2.5-flash",
        tools=TOOLS_GEMINI,
        system_instruction=SYSTEM_PROMPT
    )

    # Inicia sessão de chat (mantém histórico automaticamente)
    chat = modelo.start_chat(enable_automatic_function_calling=False)

    print("✅ Gemini conectado. Ferramentas: buscar_filmes_rag, buscar_na_web")
    print("💬 Digite sua pergunta (ou 'sair' para encerrar)\n")

    while True:
        try:
            user_input = input("👤 Você: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if user_input.lower() in ("sair", "exit", "quit"):
            break
        if not user_input:
            continue

        # ── Agentic loop: Gemini pode chamar ferramentas múltiplas vezes ──────
        mensagem = user_input
        while True:
            resposta = chat.send_message(mensagem)
            parte = resposta.candidates[0].content.parts[0]

            # Se o Gemini respondeu com texto → fim do turno
            if hasattr(parte, "text") and parte.text:
                print(f"\n🤖 Gemini: {parte.text}\n")
                break

            # Se o Gemini quer chamar uma ferramenta
            if hasattr(parte, "function_call") and parte.function_call:
                fc   = parte.function_call
                nome = fc.name
                args = dict(fc.args)

                print(f"🔧 Chamando ferramenta: {nome}({args})")
                resultado = executar_ferramenta(nome, args)
                print(f"   ↳ {resultado[:120]}{'...' if len(resultado) > 120 else ''}")

                # Devolve o resultado ao Gemini como function_response
                from google.generativeai.types import content_types
                mensagem = {
    "parts": [{
        "function_response": {
            "name": nome,
            "response": {"result": resultado}
        }
    }]
}
                # Continua o loop para o Gemini processar o resultado
                continue

            # Fallback: resposta inesperada
            print(f"\n🤖 Gemini: {resposta.text}\n")
            break

    print("\n👋 Até logo!")


if __name__ == "__main__":
    main()
