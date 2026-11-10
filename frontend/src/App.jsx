import { useState } from "react";
import { ask } from "./api";

export default function App() {
  const [message, setMessage] = useState("");
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);

  async function handleSubmit(event) {
    event.preventDefault();
    const text = message.trim();
    if (!text || loading) return;

    setMessage("");
    setMessages((prev) => [...prev, { role: "user", text }]);
    setLoading(true);

    try {
      const data = await ask(text, messages);
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          text: data.answer,
          sources: data.sources || [],
          refusal: data.refusal,
        },
      ]);
    } catch (error) {
      setMessages((prev) => [
        ...prev,
        { role: "assistant", text: `Ошибка: ${error.message}`, sources: [] },
      ]);
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="app">
      <header className="header">
        <h1>Основание</h1>
        <p>ИИ-ассистент по нормативным документам</p>
      </header>

      <section className="chat">
        {messages.length === 0 && (
          <div className="empty">
            Задайте вопрос по базе нормативных документов.
          </div>
        )}

        {messages.map((item, index) => (
          <article key={index} className={`message ${item.role}`}>
            <div className="bubble">{item.text}</div>
            {item.refusal && <div className="refusal">Обоснованный отказ</div>}
            {item.sources?.length > 0 && (
              <div className="sources">
                <strong>Источники</strong>
                {item.sources.map((source, i) => (
                  <div className="source" key={i}>
                    <span>{source.doc}</span>
                    <span>{source.clause}</span>
                  </div>
                ))}
              </div>
            )}
          </article>
        ))}

        {loading && <div className="loading">Ищу по документам…</div>}
      </section>

      <form className="composer" onSubmit={handleSubmit}>
        <input
          value={message}
          onChange={(event) => setMessage(event.target.value)}
          placeholder="Например: какая арендная плата по договору с ООО «Северный Склад»?"
        />
        <button type="submit" disabled={loading}>
          Спросить
        </button>
      </form>
    </main>
  );
}
