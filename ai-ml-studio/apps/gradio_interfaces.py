"""
Gradio Web Interfaces for AI ML Studio
Easy-to-use UI for all modules
"""

import gradio as gr
from typing import List, Tuple
from pathlib import Path


def create_dataset_generator_ui():
    """Create dataset generation interface"""

    def generate_dataset(task, dataset_type, num_samples, categories_str):
        """Generate dataset via UI"""
        try:
            from backend.modules.datasets.generator import DatasetGenerator

            generator = DatasetGenerator()
            categories = [c.strip() for c in categories_str.split(",")] if categories_str else []

            if dataset_type == "Classification":
                dataset = generator.generate_text_classification_dataset(
                    task_description=task,
                    categories=categories,
                    num_samples=num_samples,
                )
            elif dataset_type == "Conversation":
                dataset = generator.generate_conversation_dataset(
                    domain=task,
                    num_conversations=num_samples,
                )
            elif dataset_type == "Q&A":
                dataset = generator.generate_qa_dataset(
                    context_source=task,
                    num_pairs=num_samples,
                )

            # Format output
            import json

            output = json.dumps(dataset[:5], indent=2)  # Show first 5

            return f"✅ Generated {len(dataset)} samples!\n\nPreview:\n{output}"

        except Exception as e:
            return f"❌ Error: {str(e)}"

    interface = gr.Interface(
        fn=generate_dataset,
        inputs=[
            gr.Textbox(label="Task Description", placeholder="e.g., sentiment analysis of product reviews"),
            gr.Radio(["Classification", "Conversation", "Q&A"], label="Dataset Type", value="Classification"),
            gr.Slider(10, 1000, value=100, step=10, label="Number of Samples"),
            gr.Textbox(label="Categories (comma-separated, for classification)", placeholder="positive, negative, neutral"),
        ],
        outputs=gr.Textbox(label="Output", lines=20),
        title="🎲 Synthetic Dataset Generator",
        description="Generate training datasets using Claude AI",
    )

    return interface


def create_rag_ui():
    """Create RAG interface"""
    from backend.modules.rag import RAGSystem

    # Global RAG system
    rag_system = None

    def index_documents(collection_name, documents_text):
        """Index documents"""
        try:
            nonlocal rag_system
            rag_system = RAGSystem(collection_name=collection_name)

            # Split documents by lines
            documents = [doc.strip() for doc in documents_text.split("\n") if doc.strip()]

            rag_system.index_documents(documents)

            return f"✅ Indexed {len(documents)} documents into collection '{collection_name}'"

        except Exception as e:
            return f"❌ Error: {str(e)}"

    def query_rag(question, top_k):
        """Query RAG system"""
        try:
            if rag_system is None:
                return "❌ Please index documents first!"

            result = rag_system.query(question=question, top_k=top_k)

            output = f"**Answer:**\n{result['answer']}\n\n**Sources:**\n"

            for i, source in enumerate(result.get("sources", []), 1):
                output += f"\n{i}. {source['text'][:200]}... (score: {source['score']:.3f})\n"

            return output

        except Exception as e:
            return f"❌ Error: {str(e)}"

    with gr.Blocks() as interface:
        gr.Markdown("# 🔍 RAG System")

        with gr.Tab("Index Documents"):
            collection_input = gr.Textbox(label="Collection Name", value="default")
            documents_input = gr.Textbox(
                label="Documents (one per line)",
                placeholder="Enter your documents here...",
                lines=10,
            )
            index_button = gr.Button("Index Documents")
            index_output = gr.Textbox(label="Status")

            index_button.click(
                index_documents,
                inputs=[collection_input, documents_input],
                outputs=index_output,
            )

        with gr.Tab("Query"):
            question_input = gr.Textbox(label="Question", placeholder="Ask a question...")
            top_k_input = gr.Slider(1, 10, value=5, step=1, label="Number of Sources")
            query_button = gr.Button("Query")
            query_output = gr.Markdown()

            query_button.click(
                query_rag,
                inputs=[question_input, top_k_input],
                outputs=query_output,
            )

    return interface


def create_image_classifier_ui():
    """Create image classification interface"""

    def classify_image(image, model_name):
        """Classify uploaded image"""
        try:
            from backend.modules.vision.classifier import create_classifier
            from PIL import Image

            # Create classifier
            classifier = create_classifier(architecture=model_name, pretrained=True)

            # Predict
            results = classifier.predict(image)

            # Format output
            output = "**Top Predictions:**\n\n"
            for i, (class_id, prob) in enumerate(results, 1):
                output += f"{i}. Class {class_id}: {prob*100:.2f}%\n"

            return output

        except Exception as e:
            return f"❌ Error: {str(e)}"

    interface = gr.Interface(
        fn=classify_image,
        inputs=[
            gr.Image(type="pil", label="Upload Image"),
            gr.Dropdown(
                ["efficientnet_b0", "resnet50", "vit_base_patch16_224"],
                label="Model",
                value="efficientnet_b0",
            ),
        ],
        outputs=gr.Markdown(label="Predictions"),
        title="🖼️ Image Classifier",
        description="Classify images using state-of-the-art models",
    )

    return interface


def launch_all_interfaces(share=False):
    """Launch all interfaces in tabs"""

    with gr.Blocks(title="AI ML Studio") as demo:
        gr.Markdown("# 🚀 AI ML Studio - Web Interface")

        with gr.Tab("Dataset Generator"):
            create_dataset_generator_ui().render()

        with gr.Tab("RAG System"):
            create_rag_ui().render()

        with gr.Tab("Image Classifier"):
            create_image_classifier_ui().render()

    demo.launch(share=share, server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    launch_all_interfaces()
