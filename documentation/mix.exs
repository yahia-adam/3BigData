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
      authors: "Yahia abdchafee adam",
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
          "md/introduction/objectives.md": [title: "Objectives"],
          "md/application_web/overview.md": [title: "Overview"],
          "md/bibliotheque_rust/linear_model.md": [title: "Linear Model"],
          "md/bibliotheque_rust/pmc.md": [title: "Perceptron Multi Couches"],
          "md/bibliotheque_rust/rbf.md": [title: "Radial Basis Function"],
          "md/bibliotheque_rust/svm.md": [title: "Support Vector Machine"],
          "md/bibliotheque_rust/test.md": [title: "Tests"],
          "md/dataset/overview.md": [title: "Overview"],
          "md/etapes_avancement/steps.md": [title: "Steps"],
        ],
        groups_for_extras: [
          "INTRODUCTION": Path.wildcard("md/introduction/*.md"),
          "APPLICATION WEB": Path.wildcard("md/application_web/*.md"),
          "BIBLIOTHÈQUE EN RUST": Path.wildcard("md/bibliotheque_rust/*.md"),
          "DATASET": Path.wildcard("md/dataset/*.md"),
          "ÉTAPES D'AVANCEMENT": Path.wildcard("md/etapes_avancement/*.md"),
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
