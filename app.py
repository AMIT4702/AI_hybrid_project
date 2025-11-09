import streamlit as st
import os
import fitz
import pandas as pd
import re
import json
import numpy as np
import requests
from sentence_transformers import SentenceTransformer
import faiss
import duckdb
import difflib
import tempfile
import csv
import logging
from pydantic import PrivateAttr
from phi.model.base import Model
import plotly.express as px

# -------------------------------
# Logging
# -------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# -------------------------------
# API KEYS (replace with st.secrets in prod)
# -------------------------------
import streamlit as st

api_key = st.secrets["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]
# -------------------------------
# Sidebar Navigation
# -------------------------------
st.set_page_config(page_title="🚀 Unified AI Platform", layout="wide")
st.sidebar.title("🔍 What do you want to do?")
choice = st.sidebar.radio("Select a mode:", ["📊 Analysis", "📝 Summarization","Algorithm","Image Generation"])

# =====================================================
# 📊 ANALYST MODULE
# =====================================================
class OpenRouterChatWrapper(Model):
    name: str = "openrouter"
    model: str = "meta-llama/llama-3.3-70b-instruct"
    api_key: str = LLM_model

    def run(self, messages: list[dict], **kwargs) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", 0),
            "max_tokens": kwargs.get("max_tokens", 500)
        }
        resp = requests.post("https://openrouter.ai/api/v1/chat/completions",
                             headers=headers, json=payload)
        data = resp.json()
        try:
            return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            return f"Error: {e}, Response: {data}"

def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns
          .str.strip()
          .str.replace(r"\s+", "_", regex=True)
          .str.replace(r"[^A-Za-z0-9_]", "", regex=True)
          .str.replace(r"^_+", "", regex=True)
          .str.replace(r"_+", "_", regex=True)
    )
    return df

def preprocess_and_save(file):
    try:
        if file.name.endswith(".csv"):
            df = pd.read_csv(file, encoding="utf-8", na_values=["NA", "N/A", "missing"])
        elif file.name.endswith(".xlsx"):
            df = pd.read_excel(file, na_values=["NA", "N/A", "missing"])
        else:
            st.error("Unsupported file format. Please upload a CSV or Excel file.")
            return None, None, None

        df = clean_column_names(df)
        for col in df.select_dtypes(include=["object"]):
            df[col] = df[col].astype(str).replace({r'"': '""'}, regex=True)
        for col in df.columns:
            if "date" in col.lower():
                df[col] = pd.to_datetime(df[col], errors="coerce")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            temp_path = tmp.name
            df.to_csv(temp_path, index=False, quoting=csv.QUOTE_ALL)

        return temp_path, df.columns.tolist(), df
    except Exception as e:
        st.error(f"Error processing file: {e}")
        return None, None, None

def map_columns(user_question: str, columns: list[str]) -> list[str]:
    mapped_cols = []
    for word in user_question.split():
        best_match = difflib.get_close_matches(word, columns, n=1, cutoff=0.6)
        if best_match:
            mapped_cols.append(best_match[0])
    return list(set(mapped_cols))

def nl_to_sql(openrouter_model: OpenRouterChatWrapper, user_question: str, columns: list[str]) -> str:
    system = (
        "You are a senior data analyst. Generate a valid DuckDB SQL query for table `uploaded_data`. "
        "⚠️ Rules: wrap column names in double quotes; handle approximate matches; use TRIM/ILIKE as needed. "
        f"Available columns: {', '.join(columns)}"
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_question}
    ]
    text = openrouter_model.run(messages)
    m = re.search(r"```sql\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return text.strip()

# =====================================================
# 📝 SUMMARIZER MODULE
# =====================================================
performance_data = ""

def extract_text_from_pdf(file_path):
    text = ""
    try:
        doc = fitz.open(file_path)
        for page in doc:
            text += page.get_text()
    except Exception as e:
        print(f"❌ Failed PDF: {file_path} ({e})")
    return text

def extract_text_from_excel(file_path):
    text = ""
    try:
        dfs = pd.read_excel(file_path, sheet_name=None)
        for sheet_name, df in dfs.items():
            text += f"\n\n--- Sheet: {sheet_name} ---\n"
            text += df.astype(str).to_string(index=False)
    except Exception as e:
        print(f"❌ Failed Excel: {file_path} ({e})")
    return text

def clean_text(text):
    text = re.sub(r"\n+", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"Page \d+|Slide \d+", "", text)
    text = re.sub(r"http\S+|www\S+", "", text)
    text = text.encode("ascii", "ignore").decode()
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def get_all_text_from_files(uploaded_files):
    global performance_data
    all_text_chunks = []
    for uploaded_file in uploaded_files:
        ext = uploaded_file.name.lower().split(".")[-1]
        content = ""
        try:
            if ext == "pdf":
                with open(uploaded_file.name, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                content = extract_text_from_pdf(uploaded_file.name)
            elif ext in ["xls", "xlsx"]:
                with open(uploaded_file.name, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                content = extract_text_from_excel(uploaded_file.name)
            elif ext == "csv":
                df = pd.read_csv(uploaded_file)
                content = df.astype(str).to_string(index=False)
            elif ext == "txt":
                content = uploaded_file.read().decode("utf-8")
            else:
                continue
            cleaned = clean_text(content)
            if "score_analysis" in uploaded_file.name:
                performance_data = cleaned
            if cleaned.strip():
                all_text_chunks.append({"source": uploaded_file.name, "content": cleaned})
        except Exception as e:
            st.error(f"❌ Failed to process {uploaded_file.name}: {e}")
    return all_text_chunks

def chunk_text(documents, chunk_size=500, overlap=100):
    chunked_docs = []
    for doc in documents:
        words = doc["content"].split()
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            if len(chunk.strip()) > 50:
                chunked_docs.append({"source": doc["source"], "chunk": chunk})
    return chunked_docs

model_embed = SentenceTransformer("all-MiniLM-L6-v2",device="cpu")

def embed_chunks_with_cache(chunks, cache_path="embedding_cache.json"):
    if os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            cache = json.load(f)
    else:
        cache = {}
    embedded_data = []
    for chunk in chunks:
        text = chunk["chunk"]
        if text in cache:
            emb = cache[text]
        else:
            emb = model_embed.encode(text).tolist()
            cache[text] = emb
        embedded_data.append({"embedding": emb, "text": text, "source": chunk["source"]})
    with open(cache_path, "w") as f:
        json.dump(cache, f)
    return embedded_data

def build_faiss_index(embedded_chunks):
    embedding_dim = len(embedded_chunks[0]["embedding"])
    index = faiss.IndexFlatL2(embedding_dim)
    vectors = np.array([chunk["embedding"] for chunk in embedded_chunks], dtype="float32")
    index.add(vectors)
    metadata = [{"text": chunk["text"], "source": chunk["source"]} for chunk in embedded_chunks]
    return index, metadata

def search_faiss_for_query(query, model, index, metadata, top_k=3):
    query_vector = np.array(model.encode([query]), dtype="float32")
    D, I = index.search(query_vector, top_k)
    return [metadata[idx] for idx in I[0]]

def generate_answer_with_openrouter(query, context):
    headers = {"Authorization": f"Bearer {LLM_model}", "Content-Type": "application/json"}
    payload = {
        "model": "meta-llama/llama-3.1-70b-instruct",
        "messages": [
            {"role": "system", "content": "You are REVA, a helpful assistant for TracKHR..."},
            {"role": "user", "content": f"Context:\n{context}\n\nPerformance data: {performance_data}\n\nQuestion: {query}\nAnswer:"}
        ],
        "temperature": 0.2,
        "max_tokens": 250
    }
    response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)
    data = response.json()
    try:
        return data["choices"][0]["message"]["content"].strip()
    except KeyError:
        return "Sorry, I couldn’t generate a response."

# =====================================================
# MAIN APP FLOW
# =====================================================
if choice == "📊 Analysis":
    st.title("📊 AI Data Analyst")

    # ---------------------------
    # Initialize session_state
    # ---------------------------
    for k, v in {
        "query_history": [],   # ✅ store all queries here
        "uploaded_path": "",
        "columns": [],
        "df_head": None,
    }.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # ---------------------------
    # File Upload
    # ---------------------------
    uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])

    if uploaded_file is not None:
        temp_path, columns, df = preprocess_and_save(uploaded_file)

        if temp_path and columns and df is not None:
            st.session_state.uploaded_path = temp_path
            st.session_state.columns = columns

            st.subheader("Uploaded Data (cleaned columns)")
            st.dataframe(df.head(20), use_container_width=True)
            st.caption(f"Columns: {columns}")

            # Load into DuckDB
            con = duckdb.connect()
            con.execute(f"""
            CREATE OR REPLACE TABLE uploaded_data AS
            SELECT * FROM read_csv_auto(
                '{st.session_state.uploaded_path}',
                header=True,
                ignore_errors=True
            );
            """)

            st.markdown("**Preview (Top 10):**")
            st.dataframe(con.execute("SELECT * FROM uploaded_data LIMIT 10").fetchdf(),
                         use_container_width=True)

            # OpenRouter Model
            openrouter_model = OpenRouterChatWrapper(model="meta-llama/llama-3.3-70b-instruct")

            # ---------------------------
            # Query Input
            # ---------------------------
            st.markdown("---")
            user_query = st.text_area(
                "Ask a question or paste a SQL query (both work):",
                placeholder="Examples: Show me top 10 rows • Average Salary by Department • SELECT * FROM uploaded_data LIMIT 5",
                height=110,
                key="user_query_input"
            )

            # ---------------------------
            # Run Query
            # ---------------------------
            if st.button("Submit Query", type="primary"):
                if st.session_state.user_query_input.strip() == "":
                    st.warning("Please enter a query.")
                else:
                    if st.session_state.user_query_input.strip().lower().startswith("select"):
                        sql_query = st.session_state.user_query_input.strip()
                    else:
                        sql_query = nl_to_sql(
                            openrouter_model,
                            st.session_state.user_query_input,
                            st.session_state.columns
                        )

                    try:
                        result_df = con.execute(sql_query).fetchdf()
                        # ✅ Save query + result into history
                        st.session_state.query_history.append({
                            "sql": sql_query,
                            "result": result_df
                        })
                        st.success("✅ Query executed successfully!")
                    except Exception as e:
                        st.error(f"❌ SQL execution failed: {e}")

            # ---------------------------
            # Display Query History
            # ---------------------------
            if st.session_state.query_history:
                st.subheader("📜 Query History")

                for idx, q in enumerate(st.session_state.query_history, start=1):
                    st.markdown(f"**Query {idx}:**")
                    st.code(q["sql"], language="sql")
                    st.dataframe(q["result"], use_container_width=True)

                    # Visualization for each query
                    with st.expander(f"📊 Visualize Query {idx}"):
                        result_df = q["result"].copy()
                        result_df.columns = [
                            str(c).strip().replace("()", "").replace('"', "")
                            for c in result_df.columns
                        ]

                        chart_type = st.selectbox(
                            f"Choose chart type for Query {idx}:",
                            ["None", "Bar", "Line", "Area", "Pie"],
                            index=0,
                            key=f"chart_{idx}"  # unique key per query
                        )

                        if chart_type != "None":
                            cols2 = st.columns(2)
                            with cols2[0]:
                                x_col = st.selectbox(
                                    "X-axis:", result_df.columns.tolist(), key=f"xcol_{idx}"
                                )
                            with cols2[1]:
                                numeric_cols = result_df.select_dtypes(include=["number"]).columns.tolist()
                                if not numeric_cols:
                                    st.warning("No numeric columns found for Y-axis.")
                                    st.stop()
                                y_col = st.selectbox(
                                    "Y-axis (numeric):", numeric_cols, key=f"ycol_{idx}"
                                )

                            try:
                                if chart_type == "Bar":
                                    st.bar_chart(data=result_df, x=x_col, y=y_col, use_container_width=True)
                                elif chart_type == "Line":
                                    st.line_chart(data=result_df, x=x_col, y=y_col, use_container_width=True)
                                elif chart_type == "Area":
                                    st.area_chart(data=result_df, x=x_col, y=y_col, use_container_width=True)
                                elif chart_type == "Pie":
                                    fig = px.pie(result_df, names=x_col, values=y_col, title=f"{y_col} by {x_col}")
                                    st.plotly_chart(fig, use_container_width=True)
                            except Exception as e:
                                st.error(f"Graph rendering failed: {e}")

                # ---------------------------
                # Clear All Queries Button
                # ---------------------------
                if st.button("🗑️ Clear All Queries"):
                    st.session_state.query_history = []
                    st.success("History cleared.")

    else:
        st.info("Upload a CSV or Excel file to begin.")
#=========================
# Summarization part
#=========================
elif choice == "📝 Summarization":
    st.title("📝 AI Document Summarizer")
    uploaded_files = st.file_uploader("Upload documents (PDF, CSV, Excel, TXT)", type=["pdf", "csv", "xlsx", "txt"], accept_multiple_files=True)
    if uploaded_files:
        docs = get_all_text_from_files(uploaded_files)
        chunked_docs = chunk_text(docs)
        embedded_chunks = embed_chunks_with_cache(chunked_docs)
        index, metadata = build_faiss_index(embedded_chunks)

        query = st.text_area("💬 Ask a question about your documents:")
        if st.button("Submit Query"):
            results = search_faiss_for_query(query, model_embed, index, metadata, top_k=3)
            context = "\n\n".join([res["text"] for res in results])
            answer = generate_answer_with_openrouter(query, context)
            st.markdown("### 🤖 Answer")
            st.write(answer)

if choice == "Algorithm":
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.metrics import classification_report, confusion_matrix
    import matplotlib.pyplot as plt
    import seaborn as sns

    st.title("ML Algorithm Selector")

    uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])

    if uploaded_file is not None:
        # Load file
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.write("### Uploaded Dataset", df.head())

        # ---------------- Data Preprocessing ----------------
        label_encoders = {}
        for col in df.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le

        df = df.fillna(df.mean(numeric_only=True))  # handle missing values

        X = df.iloc[:, :-1]   # features
        y = df.iloc[:, -1]    # target

        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # ---------------- Algorithm Selection ----------------
        st.sidebar.title("Select algorithm")
        algo_choice = st.sidebar.radio(
            "Choose an algorithm:", 
            ["Logistic Regression", "Decision Tree", "SVM", "Random Forest", "Naive Bayes"]
        )

        if algo_choice == "Logistic Regression":
            from sklearn.linear_model import LogisticRegression
            model = LogisticRegression()

        elif algo_choice == "Decision Tree":
            from sklearn.tree import DecisionTreeClassifier
            model = DecisionTreeClassifier()

        elif algo_choice == "SVM":
            from sklearn.svm import SVC
            model = SVC()

        elif algo_choice == "Random Forest":
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier()

        elif algo_choice == "Naive Bayes":
            from sklearn.naive_bayes import GaussianNB   # ✅ fixed
            model = GaussianNB()

        # ---------------- Training ----------------
        model.fit(X_train, y_train)
        accuracy = model.score(X_test, y_test)
        st.success(f"{algo_choice} Accuracy: {accuracy:.2f}")

        # ---------------- Evaluation ----------------
        y_pred = model.predict(X_test)
        st.write("### Classification Report")
        st.text(classification_report(y_test, y_pred))

        st.write("### Confusion Matrix")
        fig, ax = plt.subplots()
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues", ax=ax)
        st.pyplot(fig)

        # ---------------- Prediction on new input ----------------
        st.write("### Make Prediction")
        input_data = []
        for col in df.columns[:-1]:
            val = st.number_input(f"Enter value for {col}", value=float(df[col].mean()))
            input_data.append(val)

        if st.button("Predict"):
            input_scaled = scaler.transform([input_data])
            prediction = model.predict(input_scaled)[0]

            # Decode if target was categorical
            if df.columns[-1] in label_encoders:
                prediction_label = label_encoders[df.columns[-1]].inverse_transform([prediction])[0]
            else:
                prediction_label = prediction

            st.success(f"Predicted Class: {prediction_label}")


if choice == "Algorithm":
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.metrics import classification_report, confusion_matrix
    import matplotlib.pyplot as plt
    import seaborn as sns

    st.title("ML Algorithm Selector")

    uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])

    if uploaded_file is not None:
        # Load file
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.write("### Uploaded Dataset", df.head())

        # ---------------- Data Preprocessing ----------------
        label_encoders = {}
        for col in df.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le

        df = df.fillna(df.mean(numeric_only=True))  # handle missing values

        X = df.iloc[:, :-1]   # features
        y = df.iloc[:, -1]    # target

        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # ---------------- Algorithm Selection ----------------
        st.sidebar.title("Select algorithm")
        algo_choice = st.sidebar.radio(
            "Choose an algorithm:", 
            ["Logistic Regression", "Decision Tree", "SVM", "Random Forest", "Naive Bayes"]
        )

        if algo_choice == "Logistic Regression":
            from sklearn.linear_model import LogisticRegression
            model = LogisticRegression()

        elif algo_choice == "Decision Tree":
            from sklearn.tree import DecisionTreeClassifier
            model = DecisionTreeClassifier()

        elif algo_choice == "SVM":
            from sklearn.svm import SVC
            model = SVC()

        elif algo_choice == "Random Forest":
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier()

        elif algo_choice == "Naive Bayes":
            from sklearn.naive_bayes import GaussianNB   # ✅ fixed
            model = GaussianNB()

        # ---------------- Training ----------------
        model.fit(X_train, y_train)
        accuracy = model.score(X_test, y_test)
        st.success(f"{algo_choice} Accuracy: {accuracy:.2f}")

        # ---------------- Evaluation ----------------
        y_pred = model.predict(X_test)
        st.write("### Classification Report")
        st.text(classification_report(y_test, y_pred))

        st.write("### Confusion Matrix")
        fig, ax = plt.subplots()
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues", ax=ax)
        st.pyplot(fig)

        # ---------------- Prediction on new input ----------------
        st.write("### Make Prediction")
        input_data = []
        for col in df.columns[:-1]:
            val = st.number_input(f"Enter value for {col}", value=float(df[col].mean()))
            input_data.append(val)

        if st.button("Predict"):
            input_scaled = scaler.transform([input_data])
            prediction = model.predict(input_scaled)[0]

            # Decode if target was categorical
            if df.columns[-1] in label_encoders:
                prediction_label = label_encoders[df.columns[-1]].inverse_transform([prediction])[0]
            else:
                prediction_label = prediction

            st.success(f"Predicted Class: {prediction_label}")
#API_KEY = "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"


