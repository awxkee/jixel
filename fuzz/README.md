# AFL++ encoder target

Install and run AFL:

```sh
cargo install cargo-afl
cd fuzz
cargo afl build --release --bin encode
cargo afl fuzz -i in -o out target/release/encode
```

Reproduce a crash by feeding it to the same instrumented binary:

```sh
cargo afl run --release --bin encode < out/default/crashes/<crash-file>
```
