"""Automated PDF report (ReportLab + Matplotlib)."""
from __future__ import annotations

import io
from datetime import datetime
from typing import Optional
from xml.sax.saxutils import escape

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet  # noqa: E402
from reportlab.lib.units import cm  # noqa: E402
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,  # noqa: E402
                                TableStyle)

INK, ACCENT, SOFT = colors.HexColor("#0b1220"), colors.HexColor("#10b981"), colors.HexColor("#eef2f7")


def _img(fig, width_cm: float) -> Image:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    w, h = fig.get_size_inches()
    return Image(buf, width=width_cm * cm, height=width_cm * cm * h / w)


def _styles():
    ss = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("t", parent=ss["Title"], fontSize=26, leading=30, textColor=INK, alignment=0),
        "sub": ParagraphStyle("s", parent=ss["Normal"], fontSize=11, textColor=colors.HexColor("#475569")),
        "h1": ParagraphStyle("h1", parent=ss["Heading1"], fontSize=16, textColor=INK, spaceBefore=14, spaceAfter=6),
        "h2": ParagraphStyle("h2", parent=ss["Heading2"], fontSize=12, textColor=INK, spaceBefore=8, spaceAfter=4),
        "p": ParagraphStyle("p", parent=ss["Normal"], fontSize=9.5, leading=14),
        "b": ParagraphStyle("b", parent=ss["Normal"], fontSize=9.5, leading=13, leftIndent=12, bulletIndent=2),
        "small": ParagraphStyle("sm", parent=ss["Normal"], fontSize=8, textColor=colors.HexColor("#64748b")),
    }


def _table(rows, col_widths=None, header=True):
    t = Table(rows, colWidths=col_widths, hAlign="LEFT")
    style = [("FONTSIZE", (0, 0), (-1, -1), 8.5), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
             ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd5e1")),
             ("ROWBACKGROUNDS", (0, 1 if header else 0), (-1, -1), [colors.white, SOFT]),
             ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
    if header:
        style += [("BACKGROUND", (0, 0), (-1, 0), INK), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                  ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold")]
    t.setStyle(TableStyle(style))
    return t


def _fmt(v, nd=3):
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.{nd}g}"
    return str(v)


def _footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#64748b"))
    canvas.drawString(2 * cm, 1.2 * cm, "ML Insight Studio - automated analysis report")
    canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"Page {doc.page}")
    canvas.restoreState()


def build_pdf(meta: dict, prep: Optional[dict], eda: dict, ml: Optional[dict], insights: Optional[dict],
              pdp: Optional[list] = None) -> bytes:
    st = _styles()
    P = lambda text, style="p": Paragraph(escape(str(text)), st[style])  # noqa: E731
    story = []

    # ---- cover / summary
    story += [Spacer(1, 1.2 * cm), Paragraph("Automated Data Analysis Report", st["title"]),
              P(f"Dataset: {meta.get('filename', 'dataset')}   |   Generated {datetime.now():%d %b %Y %H:%M}", "sub"), Spacer(1, 0.6 * cm)]
    if insights:
        story += [P("Executive summary", "h1"), P(insights["headline"])]
        for f in insights["findings"][:6]:
            story.append(Paragraph(f"<b>{escape(f['title'])}:</b> {escape(f['text'])}", st["b"], bulletText="•"))
    ov = eda["overview"]
    story += [P("1. Dataset overview", "h1"),
              _table([["Rows", "Columns", "Numeric", "Categorical", "Missing cells", "Duplicates"],
                      [ov["rows"], ov["columns"], ov["numeric"], ov["categorical"], ov["missing_cells"], ov["duplicates"]]])]
    rows = [["Column", "Type", "Missing %", "Unique", "Mean / top value"]]
    for c in eda["columns"][:25]:
        rows.append([c["name"][:28], c["kind"], f"{c['missing_pct']:.1f}", c["unique"],
                     _fmt(c.get("mean")) if c["kind"] == "numeric" else (c["top_values"][0]["value"][:20] if c.get("top_values") else "-")])
    story += [Spacer(1, 6), _table(rows)]

    # ---- preprocessing
    story.append(P("2. Data preprocessing", "h1"))
    if prep:
        s = prep["summary"]
        story.append(_table([["", "Before", "After"],
                             ["Rows", s["rows_before"], s["rows_after"]], ["Columns", s["cols_before"], s["cols_after"]],
                             ["Missing cells", s["missing_before"], s["missing_after"]],
                             ["Completeness %", s["completeness_before"], s["completeness_after"]]], [5 * cm, 4 * cm, 4 * cm]))
        story.append(Spacer(1, 6))
        for step in prep["steps"]:
            tag = "" if step["status"] == "done" else " (skipped)"
            story.append(Paragraph(f"<b>{escape(step['title'])}</b> <font color='#10b981'>[{escape(step['method'])}]</font>{tag}: {escape(step['detail'])}",
                                   st["b"], bulletText="•"))
        o = prep.get("outliers", {})
        if o.get("ran") and o.get("loss_curve"):
            fig, ax = plt.subplots(figsize=(5.5, 2.2))
            ax.plot(o["loss_curve"], color="#10b981")
            ax.set_title(f"Autoencoder training loss ({o['autoencoder_backend']})", fontsize=9)
            ax.set_xlabel("training progress", fontsize=8); ax.set_ylabel("loss", fontsize=8)
            ax.tick_params(labelsize=7)
            story += [Spacer(1, 6), _img(fig, 11)]
    else:
        story.append(P("Preprocessing was not run; the analysis uses the raw file."))

    # ---- EDA
    story.append(PageBreak())
    story.append(P(f"3. Exploratory data analysis ({eda['source']} data)", "h1"))
    for ins in eda["insights"]:
        story.append(Paragraph(escape(ins["text"]), st["b"], bulletText="•"))
    hists = list(eda["histograms"].items())[:4]
    if hists:
        fig, axes = plt.subplots(2, 2, figsize=(7, 4.4))
        for ax, (name, h) in zip(axes.ravel(), hists):
            ax.bar(range(len(h)), [b["count"] for b in h], color="#3b82f6")
            ax.set_title(name, fontsize=8); ax.set_xticks([]); ax.tick_params(labelsize=7)
        for ax in axes.ravel()[len(hists):]:
            ax.axis("off")
        fig.tight_layout()
        story += [Spacer(1, 6), _img(fig, 15)]
    cm_ = eda["correlation"]
    if cm_["matrix"]:
        k = len(cm_["columns"])
        fig, ax = plt.subplots(figsize=(5.6, 4.6))
        im = ax.imshow(np.array(cm_["matrix"], dtype=float), cmap="RdYlGn", vmin=-1, vmax=1)
        ax.set_xticks(range(k)); ax.set_yticks(range(k))
        ax.set_xticklabels([c[:12] for c in cm_["columns"]], rotation=60, ha="right", fontsize=7)
        ax.set_yticklabels([c[:12] for c in cm_["columns"]], fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.04)
        ax.set_title("Correlation matrix", fontsize=9)
        story += [Spacer(1, 6), _img(fig, 11)]

    # ---- ML
    story.append(PageBreak())
    story.append(P("4. Machine learning - Random Forest", "h1"))
    if ml:
        m = ml["metrics"]
        story.append(P(f"Task: {ml['task']}   |   Target: {ml['target']}   |   Features: {len(ml['features'])}   |   "
                       f"Train rows: {ml['n_train']:,}   |   Test rows: {ml['n_test']:,}   |   Trees: {ml['model_params']['n_estimators']}"
                       + ("   |   Hyper-parameters tuned" if ml.get("tuned") else "")))
        story.append(Spacer(1, 4))
        story.append(_table([["Metric", "Value"]] + [[k.replace("_", " ").title(), f"{v:.4f}"] for k, v in m.items()], [6 * cm, 4 * cm]))
        if ml.get("cv"):
            story.append(P(f"Cross-validation ({ml['cv']['folds']} folds, {ml['cv']['metric']}): {ml['cv']['mean']:.4f} +/- {ml['cv']['std']:.4f}", "small"))
        fi = ml["feature_importance"][:12][::-1]
        fig, ax = plt.subplots(figsize=(6.2, 3.6))
        ax.barh([f["feature"][:24] for f in fi], [f["importance"] for f in fi], color="#10b981")
        ax.set_title("Feature importance (Random Forest)", fontsize=9); ax.tick_params(labelsize=7)
        fig.tight_layout()
        story += [Spacer(1, 8), _img(fig, 12.5)]
        if ml["task"] == "classification":
            cmx = ml["confusion_matrix"]
            arr = np.array(cmx["matrix"])
            fig, ax = plt.subplots(figsize=(4.4, 3.8))
            ax.imshow(arr, cmap="Blues")
            n = len(cmx["labels"])
            ax.set_xticks(range(n)); ax.set_yticks(range(n))
            ax.set_xticklabels([l[:10] for l in cmx["labels"]], fontsize=7, rotation=45, ha="right")
            ax.set_yticklabels([l[:10] for l in cmx["labels"]], fontsize=7)
            for i in range(n):
                for j in range(n):
                    ax.text(j, i, int(arr[i, j]), ha="center", va="center", fontsize=8,
                            color="white" if arr[i, j] > arr.max() / 2 else "black")
            ax.set_xlabel("Predicted", fontsize=8); ax.set_ylabel("Actual", fontsize=8); ax.set_title("Confusion matrix", fontsize=9)
            story += [Spacer(1, 6), _img(fig, 8.5)]
            story += [Spacer(1, 4), _table([["Class", "Precision", "Recall", "F1", "Support"]] +
                                           [[p["class"][:20], f"{p['precision']:.2f}", f"{p['recall']:.2f}", f"{p['f1']:.2f}", p["support"]] for p in ml["per_class"][:12]])]
        else:
            pts = ml["actual_vs_predicted"]
            fig, ax = plt.subplots(figsize=(4.6, 4))
            ax.scatter([p["x"] for p in pts], [p["y"] for p in pts], s=8, color="#3b82f6", alpha=.6)
            lo, hi = min(p["x"] for p in pts), max(p["x"] for p in pts)
            ax.plot([lo, hi], [lo, hi], color="#ef4444", lw=1)
            ax.set_xlabel("Actual", fontsize=8); ax.set_ylabel("Predicted", fontsize=8); ax.set_title("Actual vs predicted", fontsize=9)
            story += [Spacer(1, 6), _img(fig, 9)]
        rows = [["Row", "Actual", "Predicted"] + (["Confidence"] if ml["task"] == "classification" else [])]
        for r in ml["predictions_sample"][:10]:
            rows.append([r["row"], _fmt(r["actual"]), _fmt(r["predicted"])] + ([f"{r['confidence']:.2f}"] if "confidence" in r else []))
        story += [P("Sample predictions on unseen test rows", "h2"), _table(rows)]
    else:
        story.append(P("No model has been trained yet."))

    # ---- insights
    story.append(PageBreak())
    story.append(P("5. AI insights and explainability", "h1"))
    if insights:
        for f in insights["findings"]:
            story.append(Paragraph(f"<b>{escape(f['title'])}:</b> {escape(f['text'])}", st["b"], bulletText="•"))
        if pdp:
            story.append(P("How the top features affect predictions", "h2"))
            for p in pdp[:3]:
                story.append(Paragraph(f"<b>{escape(p['feature'])}</b> {escape(p['direction'])}.", st["b"], bulletText="•"))
            fig, axes = plt.subplots(1, min(3, len(pdp)), figsize=(7.4, 2.4), squeeze=False)
            for ax, p in zip(axes[0], pdp[:3]):
                if p["type"] == "number":
                    ax.plot(p["x"], p["y"], color="#10b981", marker="o", ms=3)
                else:
                    ax.bar(range(len(p["x"])), p["y"], color="#3b82f6")
                    ax.set_xticks(range(len(p["x"]))); ax.set_xticklabels([str(x)[:8] for x in p["x"]], fontsize=6, rotation=40, ha="right")
                ax.set_title(p["feature"][:18], fontsize=8); ax.tick_params(labelsize=6)
            fig.tight_layout()
            story += [Spacer(1, 4), _img(fig, 16)]
        story.append(P("Recommended next steps", "h2"))
        for n in insights["next_steps"]:
            story.append(Paragraph(escape(n), st["b"], bulletText="•"))
    else:
        story.append(P("Train a model to unlock insights."))

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm, bottomMargin=2 * cm,
                            title="Automated Data Analysis Report", author="ML Insight Studio")
    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
    return buf.getvalue()
