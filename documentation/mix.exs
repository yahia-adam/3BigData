defmodule BigDataDocumentation.MixProject do
  use Mix.Project

  def project do
    [
      app: :"3BigData_documentation",
      version: "1.0.0",
      elixir: "1.16.2",
      start_permanent: Mix.env() == :prod,
      deps: deps(),

      # Docs
      name: "Documentation",
      homepage_url: "https://3BigDatalinux.org/documentation/",
      authors: "Arnaud Cormier & Kevin Chevreuil",
      docs: [
        main: "documentation",
        api_reference: false,
        logo: "priv/assets/3BigData-logo.png",
        assets: "priv/assets",
        before_closing_head_tag: &docs_before_closing_head_tag/1,
        formatters: ["html"],
        extra_section: "Guides",
        extras: [
          "md/documentation.md": [ title: "3BigData | Documentation"],
          "md/introduction/overview.md": [title: "Overview"],
          "md/introduction/objectives.md": [title: "Objectives"],
          "md/introduction/summary.md": [title: "Summary"],
          "md/installation_et_configuration/prerequisites.md": [title: "Prerequisites"],
          "md/installation_et_configuration/cloning_repo.md": [title: "Cloning Repository"],
          "md/installation_et_configuration/installing_dependencies.md": [title: "Installing Dependencies"],
          "md/installation_et_configuration/running_application.md": [title: "Running Application"],
          "md/application_web/overview.md": [title: "Overview"],
          "md/application_web/setup.md": [title: "Setup"],
          "md/application_web/features.md": [title: "Features"],
          "md/application_web/deployment.md": [title: "Deployment"],
          "md/bibliotheque_rust/overview.md": [title: "Overview"],
          "md/bibliotheque_rust/linear_model.md": [title: "Linear Model"],
          "md/bibliotheque_rust/pmc.md": [title: "Perceptron Multi Couches"],
          "md/bibliotheque_rust/rbf.md": [title: "Radial Basis Function"],
          "md/bibliotheque_rust/svm.md": [title: "Support Vector Machine"],
          "md/algorithmes_modeles/overview.md": [title: "Overview"],
          "md/algorithmes_modeles/linear_models.md": [title: "Linear Models"],
          "md/algorithmes_modeles/neural_networks.md": [title: "Neural Networks"],
          "md/algorithmes_modeles/advanced_models.md": [title: "Advanced Models"],
          "md/dataset/overview.md": [title: "Overview"],
          "md/dataset/data_collection.md": [title: "Data Collection"],
          "md/dataset/data_preprocessing.md": [title: "Data Preprocessing"],
          "md/dataset/data_usage.md": [title: "Data Usage"],
          "md/etapes_avancement/step1.md": [title: "Step 1"],
          "md/etapes_avancement/step2.md": [title: "Step 2"],
          "md/etapes_avancement/step3.md": [title: "Step 3"],
          "md/etapes_avancement/step4.md": [title: "Step 4"],
          "md/remarques_observations/overview.md": [title: "Overview"],
          "md/remarques_observations/analysis.md": [title: "Analysis"],
          "md/remarques_observations/issues.md": [title: "Issues"],
          "md/remarques_observations/future_work.md": [title: "Future Work"],
          "md/exemples_utilisation/overview.md": [title: "Overview"],
          "md/exemples_utilisation/python_examples.md": [title: "Python Examples"],
          "md/exemples_utilisation/cpp_examples.md": [title: "C++ Examples"],
          "md/exemples_utilisation/nodejs_examples.md": [title: "Node.js Examples"],
          "md/annexes/references.md": [title: "References"],
          "md/annexes/glossary.md": [title: "Glossary"],
          "md/annexes/faq.md": [title: "FAQ"],
          "md/annexes/additional_resources.md": [title: "Additional Resources"],
          "md/contributions/guide.md": [title: "Contribution Guide"],
          "md/contributions/contributors.md": [title: "Contributors"],
          "md/contributions/code_of_conduct.md": [title: "Code of Conduct"]
        ],
        groups_for_extras: [
          "INTRODUCTION": Path.wildcard("md/introduction/*.md"),
          "INSTALLATION ET CONFIGURATION": Path.wildcard("md/installation_et_configuration/*.md"),
          "APPLICATION WEB": Path.wildcard("md/application_web/*.md"),
          "BIBLIOTHÈQUE EN RUST": Path.wildcard("md/bibliotheque_rust/*.md"),
          "ALGORITHMES ET MODÈLES": Path.wildcard("md/algorithmes_modeles/*.md"),
          "DATASET": Path.wildcard("md/dataset/*.md"),
          "ÉTAPES D'AVANCEMENT": Path.wildcard("md/etapes_avancement/*.md"),
          "REMARQUES ET OBSERVATIONS": Path.wildcard("md/remarques_observations/*.md"),
          "EXEMPLES D'UTILISATION": Path.wildcard("md/exemples_utilisation/*.md"),
          "ANNEXES": Path.wildcard("md/annexes/*.md"),
          "CONTRIBUTIONS": Path.wildcard("md/contributions/*.md")
        ],

      ],
    ]
  end

  # Run "mix help compile.app" to learn about applications.
  def application do
    [
      extra_applications: [:logger]
    ]
  end

  # Run "mix help deps" to learn about dependencies.
  defp deps do
    [
      {:ex_doc, "~> 0.31.0", only: :dev, runtime: false},
    ]
  end

  # adding custom stylesheet
  defp docs_before_closing_head_tag(:html) do
    ~s{<link rel="stylesheet" href="assets/doc.css">}
  end
  defp docs_before_closing_head_tag(_), do: ""
end
