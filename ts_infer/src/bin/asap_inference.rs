use std::{f32::consts::PI, io::stdin};

use burn::tensor;
use ndarray::prelude::*;
use burn::backend::NdArray;
use ts_infer::asap::Model;

type Backend = NdArray<f32>;

fn mov(ts: &Array1<f32>, w: usize, s: usize) -> (Array1<f32>, Array1<f32>) {
    let x: Array1<f32> = (s.max(w)..(ts.len() + s - 1))
        .step_by(s)
        .into_iter()
        .map(|x| x as f32).collect();
    let mut y = vec![];
    for ix in (s.max(w)..(ts.len() + s - 1).min(ts.len())).step_by(s) {
        let start = if ix < w + 1 { 0 } else { ix - w - 1};
        let end = ix;
        let slice = ts.slice(s![start..end]);
        y.push(slice.mean().unwrap());
    }
    return (x, Array1::from(y));
}

fn roughness(ts: &Array1<f32>) -> f32 {
    let mut answer = 0.0;
    let mut prev = ts[0];
    for e in ts.iter() {
        answer += (e - prev) * (e - prev);
        prev = *e;
    }
    return answer / (ts.len() * ts.len()) as f32;
}

fn asap(ts: &Array1<f32>, chart_w: usize, pixel_p: usize, max_drop:f32, k_sim: f32) -> (usize, usize) {
    let maxp = chart_w as f32 / pixel_p as f32;
    let slide = (ts.len() as f32 / maxp) as usize;
    let min_window = slide;
    let max_window = (ts.len() as f32 * max_drop) as i32;
    let k = kurtosis(&ts);
    let mut answer = (
        min_window, 
        slide.min(min_window), 
        roughness(&mov(&ts, min_window, slide.min(min_window)).0)
    );
    for w in (min_window..max_window as usize).step_by(1).rev() {
        let s = slide.min(w as usize);
        let (_, y) = mov(&ts, w, s);
        let mov_k = kurtosis(&y); 
        if (mov_k - k).abs() / k < k_sim {
            continue;
        }
        let mov_r = roughness(&ts);
        if answer.2 < mov_r {
            continue;
        }
        answer = (w, s, mov_r);
    }
    return (answer.0, answer.1)
}

fn kurtosis(ts: &Array1<f32>) -> f32 {
    let mut answer = 0.0;
    let mean = ts.mean().unwrap();
    let std = ts.std(0.0);
    for e in ts.iter() {
        answer += ((e - mean) / std).powf(4.0);
    }
    return answer / ts.len() as f32;
}

fn max_min_norm(ts: &Array1<f32>) -> Array1<f32> {
    let mut max = ts[0];
    let mut min = ts[0];
    for e in ts.iter() {
        max = max.max(*e);
        min = min.min(*e);
    }
    return (ts.clone() * 2.0 - max - min) / (max - min); 
}

fn gramian(ts: &Array1<f32>) -> Array2<f32> {
    let mut answer = Array2::zeros((ts.len(), ts.len()));
    for i in 0..ts.len() {
        let mut row = answer.row_mut(i);
        for j in 0..ts.len() {
            row[j] = (ts[i].acos() + PI - ts[j].acos()).cos();
        }
    }
    return answer;
}

fn mat_to_rgb(m: Array2<f32>) -> Array2<Array1<i8>> {
    let e = m[[0,0]];
    let max = m.iter().fold(e, |s, x| s.max(*x));
    let min = m.iter().fold(e, |s, x| s.min(*x));
    let norm = m.mapv(|x| (x - min)/ (max - min));
    let source = Array1::from(vec![23., 9., 138.]);
    let target = Array1::from(vec![240., 249., 32.]);
    let dx = target - source.clone();
    return norm
        .map(|x| dx.clone() * (*x) + source.clone())
        .map(|x| x.mapv( |x| x as i8));
}

fn rgb_to_tensor(m: &Array2<Array1<i8>>, size: usize) -> tensor::Tensor::<NdArray<f32>, 4> {
    let osize = m.shape()[0] as f32;
    let fsize = size as f32;
    let ratio = osize / fsize;
    let mut resized = vec![];
    for x in 0..size {
        let mut row = vec![];
        for y in 0..size {
            let i = (x as f32 * ratio) as usize;
            let j = (y as f32 * ratio) as usize;
            let rgb = m[[i , j]].clone();
            row.push(rgb); 
        }
        resized.push(row);
    }
    let mut tv = [[[[0.0; 128]; 128]; 3]; 1];
    for c in 0..3 {
        for x in 0..size {
            for y in 0..size {
                let bit = resized[x][y][c] as f32 / 255.0;
                tv[0][c][x][y] = bit;
            }
        }
    }
    //let nda = Array4::from(vec![t_v]);
    return tensor::Tensor::<NdArray<f32>, 4>::from_data(tv);
}

fn main() {
    let model: Model<Backend> = Model::default();
    let mut ts_v = vec![];
    for result in stdin().lines() {
        let line = result.expect("read line form stdin");
        if line.is_empty() {
            continue;
        }
        ts_v.push(line.parse::<f32>().unwrap());
    }
    for _ in 1..100 {
        let ts = Array1::from(ts_v.clone());
        let (w, s) = asap(&ts, 350, 3, 0.05, 0.3);
        let (_, y) = mov(&ts, w, s);
        let norm = max_min_norm(&y);
        let gaf = gramian(&norm);
        let rgb = mat_to_rgb(gaf);
        let input = rgb_to_tensor(&rgb, 128);
        let output = model.forward(input);
        println!("{:?}", output);
    }
}