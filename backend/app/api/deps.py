# backend/app/api/deps.py
from fastapi import HTTPException, Request


def get_retriever(request: Request):
    # Безопасное получение ретривера через getattr
    retriever = getattr(request.app.state, "retriever", None)

    if retriever is None:
        error_msg = getattr(
            request.app.state,
            "index_error",
            "База знаний не проиндексирована"
        )
        raise HTTPException(
            status_code=503,
            detail={
                "message": "База знаний не проиндексирована",
                "hint": "Запустите python scripts/build_index.py",
                "error": str(error_msg),
            },
        )
    return retriever