// The six stages of the pipeline (same wording as the reference design).
export const STEPS = [
  { id: 1, icon: "↥", path: "/upload", key: "upload", title: "Dataset Upload", desc: "Upload CSV or Excel dataset through React.js frontend.", tone: "navy" },
  { id: 2, icon: "✦", path: "/preprocessing", key: "preprocess", title: "Data Preprocessing", desc: "Clean, transform, validate and prepare the dataset.", tone: "forest" },
  { id: 3, icon: "▥", path: "/eda", key: "eda", title: "Exploratory Data Analysis (EDA)", desc: "Generate statistical summaries and visualizations.", tone: "navy" },
  { id: 4, icon: "◈", path: "/machine-learning", key: "ml", title: "Machine Learning", desc: "Train Random Forest model and generate predictions.", tone: "forest" },
  { id: 5, icon: "✳", path: "/insights", key: "insights", title: "AI Insights and Explainability", desc: "Explain results, feature importance and answer questions.", tone: "navy" },
  { id: 6, icon: "↗", path: "/report", key: "report", title: "Automated Report Generation", desc: "Generate downloadable PDF analysis reports.", tone: "forest" },
];
