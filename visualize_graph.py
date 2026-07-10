"""
Fichier : visualize_graph.py
Génère un unique PNG avec les deux sous-graphes côte à côte.
  - Gauche  : graph_T (Phase normative)
  - Droite  : graph_F (Phase vérification)
Output : architecture_neuro_symbolique.png
"""

import io
from PIL import Image
import matplotlib.pyplot as plt

from agent.graph import graph_T, graph_F


def get_png_bytes(graph) -> bytes | None:
    try:
        return graph.get_graph().draw_mermaid_png()
    except Exception as e:
        print(f"⚠️  PNG impossible : {e}")
        return None


def get_mermaid(graph) -> str:
    try:
        return graph.get_graph().draw_mermaid()
    except Exception as e:
        return f"# Erreur Mermaid : {e}"


def main():
    print("=" * 60)
    print("🏗️  VISUALISATION — ARCHITECTURE DEUX PHASES (PNG unique)")
    print("=" * 60)

    png_T = get_png_bytes(graph_T)
    png_F = get_png_bytes(graph_F)

    if png_T and png_F:
        img_T = Image.open(io.BytesIO(png_T))
        img_F = Image.open(io.BytesIO(png_F))

        fig, axes = plt.subplots(1, 2, figsize=(20, 12))
        fig.patch.set_facecolor("#0f1117")

        for ax, img, title in zip(
            axes,
            [img_T, img_F],
            [
                "Phase T — Construction normative\n(intent_T → universe → rules → mapping → graph)",
                "Phase F — Vérification causale\n(intent_F → projection → verification ⟳ → propagation → score)",
            ],
        ):
            ax.imshow(img)
            ax.axis("off")
            ax.set_title(
                title,
                fontsize=13,
                fontweight="bold",
                color="white",
                pad=12,
            )

        plt.suptitle(
            "Agent Neuro-Symbolique — Architecture LangGraph deux phases",
            fontsize=16,
            fontweight="bold",
            color="white",
            y=1.01,
        )

        plt.tight_layout()
        output = "architecture_neuro_symbolique.png"
        plt.savefig(output, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close()
        print(f"✅ '{output}' exporté avec succès.")

    else:
        # ── Fallback Mermaid ───────────────────────────────
        print("\n📄 Fallback : export des codes Mermaid bruts.\n")
        for graph, name in [(graph_T, "graph_T"), (graph_F, "graph_F")]:
            code = get_mermaid(graph)
            fname = f"architecture_{name}.mmd"
            with open(fname, "w", encoding="utf-8") as f:
                f.write(code)
            print(f"   → '{fname}' sauvegardé. Colle sur https://mermaid.live/")


if __name__ == "__main__":
    main()