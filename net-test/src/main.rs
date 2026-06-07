use candle_core::{Device, Tensor};
use candle_onnx;
use std::collections::HashMap;

fn main() -> anyhow::Result<()> {
    let device = Device::Cpu;

    let model_nonzero = candle_onnx::read_file("nonzero.onnx")?;

    let raw_data = vec![
        0.0f32, 1.2f32, 0.1f32, 0.3f32, 0.8f32, 0.0f32, 2.5f32, 0.0f32, 1.1f32,
    ];
    let input_data = Tensor::from_vec(raw_data, (3, 3), &device)?;

    println!("Input:");
    println!("{}\n", input_data);

    let mut inputs = HashMap::new();
    inputs.insert("input_tensor".to_string(), input_data);

    let outputs = candle_onnx::simple_eval(&model_nonzero, inputs)?;
    let indices_tensor = outputs.get("nonzero_indices").expect("Output not found");

    println!("Output:");
    println!("Dims: {:?}", indices_tensor.dims());

    println!("Nonzero indices:");
    let indices_matrix = indices_tensor.to_vec2::<i64>()?;
    for coord in indices_matrix {
        println!("  {:?}", coord);
    }

    let model_dequantize = candle_onnx::read_file("dequant.onnx")?;

    let x = Tensor::from_vec(
        vec![128u8, 129u8, 127u8, 255u8, 0u8, 200u8],
        (2, 3),
        &device,
    )?;

    println!("\nDequantize-linear -> input x (uint8):");
    println!("{:?}\n", x.to_vec2::<u8>()?);

    let mut inputs = HashMap::new();
    inputs.insert("x".to_string(), x);

    let result = candle_onnx::simple_eval(&model_dequantize, inputs)?;
    let y = result.get("y").expect("Output not found");
    println!("Dequantized-linear output: {:?}\n", y.to_vec2::<f32>()?);

    let model_dequantize_axis = candle_onnx::read_file("dequant_per_axis.onnx")?;

    let x = Tensor::from_vec(
        vec![
            3u8, 89, 34, 200, 74, 59, 5, 24, 24, 87, 32, 13, 245, 99, 4, 142, 121, 102,
        ],
        (1, 3, 3, 2),
        &device,
    )?;

    println!("\nDequantize-per-axis -> input x (uint8):");
    println!("{:?}\n", x.flatten_all()?.to_vec1::<u8>()?);

    let mut inputs = HashMap::new();
    inputs.insert("x".to_string(), x);

    let result = candle_onnx::simple_eval(&model_dequantize_axis, inputs)?;
    let y = result.get("y").expect("Output not found");
    println!(
        "Dequantized-per-axis output: {:?}\n",
        y.flatten_all()?.to_vec1::<f32>()?
    );

    let model_dequantize_blocked = candle_onnx::read_file("dequant_blocked.onnx")?;

    let x = Tensor::from_vec(
        vec![
            3u8, 89, 34, 200, 74, 59, 5, 24, 24, 87, 32, 13, 5, 12, 12, 33, 65, 42, 245, 99, 4,
            142, 121, 102,
        ],
        (1, 4, 3, 2),
        &device,
    )?;

    println!("\nDequantize-blocked -> input x (uint8):");
    println!("{:?}\n", x.flatten_all()?.to_vec1::<u8>()?);

    let mut inputs = HashMap::new();
    inputs.insert("x".to_string(), x);

    let result = candle_onnx::simple_eval(&model_dequantize_blocked, inputs)?;
    let y = result.get("y").expect("Output not found");
    println!(
        "Dequantized-blocked output: {:?}\n",
        y.flatten_all()?.to_vec1::<f32>()?
    );

    Ok(())
}
