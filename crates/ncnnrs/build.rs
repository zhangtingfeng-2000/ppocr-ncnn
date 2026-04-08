use std::env;
use std::path::{Path, PathBuf};

fn main() {
    let out_path = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR not set"));
    let include_dir = env::var("NCNN_INCLUDE_DIR")
        .map(PathBuf::from)
        .expect("ERROR: set NCNN_INCLUDE_DIR to the ncnn include directory");

    if !include_dir.join("c_api.h").exists() {
        panic!(
            "ERROR: NCNN_INCLUDE_DIR={} does not contain c_api.h",
            include_dir.display()
        );
    }

    println!("cargo:rerun-if-env-changed=NCNN_INCLUDE_DIR");
    println!("cargo:rerun-if-env-changed=NCNN_LIB_DIR");

    let bindings = bindgen::Builder::default()
        .header(format!("{}/gpu.h", include_dir.display()))
        .header(format!("{}/c_api.h", include_dir.display()))
        .clang_arg("-x")
        .clang_arg("c++")
        .allowlist_type("regex")
        .allowlist_function("ncnn.*")
        .allowlist_var("NCNN.*")
        .allowlist_type("ncnn.*")
        .opaque_type("std::vector.*")
        .wrap_unsafe_ops(true)
        .generate()
        .expect("Unable to generate bindings");

    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings");

    let lib_dir = env::var("NCNN_LIB_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| derive_lib_dir(&include_dir));
    if !lib_dir.join("ncnn.lib").exists() {
        panic!(
            "ERROR: NCNN_LIB_DIR={} does not contain ncnn.lib; set NCNN_LIB_DIR explicitly",
            lib_dir.display()
        );
    }

    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rustc-link-lib=static=ncnn");
    println!("cargo:rustc-link-lib=static=glslang");

}

fn derive_lib_dir(include_dir: &Path) -> PathBuf {
    include_dir
        .parent()
        .and_then(Path::parent)
        .map(|root| root.join("lib"))
        .unwrap_or_else(|| include_dir.join("..").join("..").join("lib"))
}
