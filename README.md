# jixel

A tiny JPEG XL encoder in Rust, ported and reworked from
[libjxl/libjxl-tiny](https://github.com/libjxl/libjxl-tiny).

## Example

```rust
fn main() {
    let output = "encoded_lossy2.jxl";
    let image = image::open(Path::new("./assets/abstract_alpha.png")).unwrap();
    let bytes = jixel::encode_image_with_alpha(
        image.to_rgba8().as_raw(),
        image.width() as usize,
        image.height() as usize,
        quality_to_distance(99.),
    );
    std::fs::write(&output, &bytes).expect("failed to write output");
}
```

## License

This project is licensed under either of

- BSD-3-Clause License (see [LICENSE](LICENSE.md))
- Apache License, Version 2.0 (see [LICENSE](LICENSE-APACHE.md))

at your option.