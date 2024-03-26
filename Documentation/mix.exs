defmodule Documentation.MixProject do
  use Mix.Project

  def project do
    [
      app: "3BigDataDocumentation",
      version: "1.1.0",
      elixir: "~> 1.14.3",
      start_permanent: Mix.env() == :prod,
      deps: deps(),

      # Docs
      name: "Documentation",
      homepage_url: "https://3bigData.com",
      authors: "Yahia abdchafee Adam",
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
          
          "md/3BigData/3bigdata-official-links.md": [ title: "Official 3BigData links"],
          "md/3BigData/why-3bigdata.md": [ title: "Why 3BigData ?"],
          
          "md/team/maintainer.md": [ title: "Maintainer"],
        ],
        groups_for_extras: [
          "3BigData": Path.wildcard("md/3BigData/*.md"),
          "APP": Path.wildcard("md/app//*.md"),
          "LIB": Path.wildcard("md/lib/*.md"),
          "LINEAR MODEL": Path.wildcard("md/linear_model/*.md"),
	        "MULTILAYER PERCEPTRON": Path.wildcard("md/multilayer_perceptron/*.md"),
          "RADICAL BASIS FUNCTION NETWORK": Path.wildcard("md/radical_basis_function_network/*.md"),
          "SUPPORT VECTOR MACHINE": Path.wildcard("md/support_vector_machine/*.md"),
          "TUTORIALS": Path.wildcard("md/tutorials/*.md"),
          "3BIG DATA TEAM": Path.wildcard("md/team/*.md"),
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
      {:ex_doc, "~> 0.29.1", only: :dev, runtime: false},
    ]
  end

  # adding custom stylesheet
  defp docs_before_closing_head_tag(:html) do
    ~s{<link rel="stylesheet" href="assets/doc.css">}
  end
  defp docs_before_closing_head_tag(_), do: ""
end
