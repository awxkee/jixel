# AFL++ fuzz targets

Two targets:

| target      | input              | exercises                                           |
|-------------|--------------------|-----------------------------------------------------|
| `encode`    | synthesized pixels | the lossy and lossless pixel encoders               |
| `transcode` | JPEG files         | the JPEG parser and lossless JPEG -> JXL transcoder |

`transcode` takes its input as a JPEG file, so it is seeded from `in_jpeg/`
rather than `in/`. Those seeds cover the distinct decode paths — baseline,
progressive, all three subsampling factors, restart intervals, optimized
Huffman tables and grayscale — which is what gives AFL somewhere to start.
Beyond checking that nothing panics, the target validates the container it gets
back: box lengths must tile the output exactly, and both a `jbrd` box and a
codestream box must be present.


Install and run AFL:

```sh
cargo install cargo-afl
cd fuzz
cargo afl build --release --bin encode

RUSTFLAGS="-Zsanitizer=address" \
cargo +nightly afl build \
  -Zbuild-std \
  --target aarch64-apple-darwin \
  --release \
  --bin encode
  
cargo afl fuzz -i in -o out target/release/encode
```

For the transcode target, generate the seed corpus first (it is not committed),
then build and run it the same way:

```sh
./make_jpeg_seeds.sh
cargo afl build --release --bin transcode
cargo afl fuzz -i in_jpeg -o out_jpeg target/release/transcode
```

On macOS AFL needs shared memory configured once, otherwise it stops with
`shmget() failed`:

```sh
cargo afl system-config   # prompts for sudo
```

Reproduce a crash by feeding it to the same instrumented binary:

```sh
cargo afl run --release --bin encode < out/default/crashes/<crash-file>
cargo afl run --release --bin transcode < out_jpeg/default/crashes/<crash-file>
```
