import streamlit as st
from retrieval import get_compliance_chain

st.set_page_config(
    page_title="ISO 27001 Auditor AI",
    layout="centered"
)
with st.sidebar:
    st.title("Auditor Dashboard")
    st.info("Grounding Data: ISO/IEC 27001:2022 Standard")
    st.warning("Note: This AI uses Reranking for high-precision compliance answers.")
    if st.button("Clear chat history"):
        st.session_state.messages = []
        st.rerun()

st.title("ISO 27001 Compliance Assistant")
st.caption("Ask me anything about the ISO 27001:2022 framework")

@st.cache_resource
def load_chain():
    return get_compliance_chain()

compliance_chain = load_chain()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a compliance question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Consulting the ISO 27001 standard..."):
            try:
                result = compliance_chain.invoke(prompt)

                answer = result.get("answer")
                sources = result.get("sources", [])

                st.markdown(answer)

                if sources:
                    with st.expander("View Auditor Sources"):
                        for i, doc in enumerate(sources):
                            page = doc.metadata.get("page", "N/A")
                            st.write(f"**Source {i+1} (Page {page}):**")
                            st.caption(doc.page_content)
                            st.divider()

                st.session_state.messages.append({"role": "assistant", "content": answer})

            except Exception as e:
                error_message = f"I encountered an error while processing your request. Please try again later."
                st.error(error_message)
                
                print(f"DEBUG ERROR: {str(e)}")
