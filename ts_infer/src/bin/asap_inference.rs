use burn::tensor;
use burn::backend::NdArray;

use burn_core::tensor::Tensor;
use ts_infer::asap::Model;

type Backend = NdArray<f32>;

fn std<const T: usize>(ts: Tensor::<Backend, T>) -> f32 {
    let mean = ts.clone().mean().into_scalar();
    let delta = (ts.clone().sub_scalar(mean)).sum().into_scalar();
    return delta * delta / ts.clone().shape().num_elements() as f32;
}
fn normalize<const T: usize>(ts: Tensor::<Backend, T>) -> Tensor::<Backend, T> {
    let mean = ts.clone().mean().into_scalar();
    let std = std(ts.clone());
    return (ts.clone().sub_scalar(mean)) / std;
}

fn main() {
    let model: Model<Backend> = Model::default();
    for _ in 1..100 {
        let input = tensor::Tensor::<NdArray<f32>, 4>::zeros([1, 3, 128, 128]);
        let output = model.forward(input);
        println!("{:?}", output);
    }
}