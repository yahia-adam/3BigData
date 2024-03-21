Cargo uses conventions for file placement to make it easy to dive into a new Cargo package:
```.
├── Cargo.lock
├── Cargo.toml
├── src/
│   ├── lib.rs
│   ├── main.rs
│   └── bin/
│       ├── named-executable.rs
│       ├── another-executable.rs
│       └── multi-file-executable/
│           ├── main.rs
│           └── some_module.rs
├── benches/
│   ├── large-input.rs
│   └── multi-file-bench/
│       ├── main.rs
│       └── bench_module.rs
├── examples/
│   ├── simple.rs
│   └── multi-file-example/
│       ├── main.rs
│       └── ex_module.rs
└── tests/
    ├── some-integration-tests.rs
    └── multi-file-test/
        ├── main.rs
        └── test_module.rs
```

- ``` Cargo.toml ``` and ``` Cargo.lock ``` are stored in the root of your package (package root).
- Source code goes in the src directory.
- The default library file is ``` src/lib.rs ```.
- The default executable file is ``` src/main.rs```.
    - Other executables can be placed in ```src/bin/```.
- Benchmarks go in the ```benches``` directory.
- Examples go in the ```examples``` directory.
- Integration tests go in the ```tests``` directory.
