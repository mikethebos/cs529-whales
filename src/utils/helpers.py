def conv2d_output_dim(dim_in: int, kernel_dim: int, stride: int = 1, padding: int = 0, dilation: int = 1) -> int:
    """
    Get output height or width for a Conv2d layer, 
    where total output shape is (batch_size, out_channels, out_height, out_width)
    :param dim_in: int, incoming height or width
    :param kernel_dim: int, height or width of filter
    :param stride: int, distance to move filter at each step
    :param padding: int, extra entries added to dim_in
    :param dilation: int, distance between start of a filter's cell and the next cell
    :return out, int, the output height or width of the Conv2d layer
    """
    return 1 + ((dim_in + (2 * padding) - (dilation * (kernel_dim - 1)) - 1) // stride)