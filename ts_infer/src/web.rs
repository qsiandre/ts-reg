#![allow(clippy::new_without_default)]

use burn::backend::NdArray;
use js_sys::Array;

#[cfg(target_family = "wasm")]
use wasm_bindgen::prelude::*;

use crate::asap::Model;

use burn::tensor::Tensor;

use crate::ts;
#[cfg_attr(target_family = "wasm", wasm_bindgen(start))]
pub fn start() {
    console_error_panic_hook::set_once();
}

type Backend = NdArray<f32>;
/// Mnist structure that corresponds to JavaScript class.
/// See:[exporting-rust-struct](https://rustwasm.github.io/wasm-bindgen/contributing/design/exporting-rust-struct.html)
#[cfg_attr(target_family = "wasm", wasm_bindgen)]
pub struct Asap {
    model: Option<Model<Backend>>,
}

#[cfg_attr(target_family = "wasm", wasm_bindgen)]
impl Asap {
    #[cfg_attr(target_family = "wasm", wasm_bindgen(constructor))]
    pub fn new() -> Self {
        console_error_panic_hook::set_once();
        Self { model: None }
    }

    pub async fn inference(&mut self, input: &[f32]) -> Result<Array, String> {
        if self.model.is_none() {
            let model: Model<NdArray<f32>> = Model::default();
            self.model = Some(model);
        }

        let model = self.model.as_ref().unwrap();

        let input: Tensor<Backend, 4> = ts::to_tensor(input); 
        let output: Tensor<Backend, 2> = model.forward(input);

        #[cfg(not(target_family = "wasm"))]
        let output = output.into_data().convert::<f32>().value;

        #[cfg(target_family = "wasm")]
        let output = output.into_data().await.convert::<f32>().value;

        let array = Array::new();
        for value in output {
            array.push(&value.into());
        }

        Ok(array)
    }

    pub async fn gramian_rgb(&mut self, input: &[f32]) -> Result<Array, String> {
        let rgb = ts::to_rgb(input);
        let array = Array::new();
        array.push(&128.into());
        array.push(&128.into());
        for i in 0..128 {
            for j in 0..128 {
                for c in 0..3 {
                    array.push(&rgb[i][j][c].into());
                }
            }
        }
        Ok(array)
    }

    pub async fn smooth(&mut self, input: &[f32]) -> Result<Array, String> {
        let (x, y) = ts::smooth(input);
        let array = Array::new();
        let s = x.shape();
        for i in 0..s[0] {
            array.push(&x[i].into());
            array.push(&y[i].into());
        }
        Ok(array)
    }
}