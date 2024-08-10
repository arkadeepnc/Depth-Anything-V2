import torch
import warnings
from typing import List, NamedTuple, Optional, Tuple, Union, Sequence

# https://github.com/facebookresearch/pytorch3d/tree/main/pytorch3d/ops
# https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/ops/points_alignment.py
AMBIGUOUS_ROT_SINGULAR_THR = 1e-15

class SimilarityTransform(NamedTuple):
    R: torch.Tensor
    T: torch.Tensor
    s: torch.Tensor

def list_to_padded(
    x: Union[List[torch.Tensor], Tuple[torch.Tensor]],
    pad_size: Union[Sequence[int], None] = None,
    pad_value: float = 0.0,
    equisized: bool = False,
) -> torch.Tensor:
    r"""
    Transforms a list of N tensors each of shape (Si_0, Si_1, ... Si_D)
    into:
    - a single tensor of shape (N, pad_size(0), pad_size(1), ..., pad_size(D))
      if pad_size is provided
    - or a tensor of shape (N, max(Si_0), max(Si_1), ..., max(Si_D)) if pad_size is None.
    Args:
      x: list of Tensors
      pad_size: list(int) specifying the size of the padded tensor.
        If `None` (default), the largest size of each dimension
        is set as the `pad_size`.
      pad_value: float value to be used to fill the padded tensor
      equisized: bool indicating whether the items in x are of equal size
        (sometimes this is known and if provided saves computation)
    Returns:
      x_padded: tensor consisting of padded input tensors stored
        over the newly allocated memory.
    """
    if equisized:
        return torch.stack(x, 0)

    if not all(torch.is_tensor(y) for y in x):
        raise ValueError("All items have to be instances of a torch.Tensor.")

    # we set the common number of dimensions to the maximum
    # of the dimensionalities of the tensors in the list
    element_ndim = max(y.ndim for y in x)

    # replace empty 1D tensors with empty tensors with a correct number of dimensions
    x = [
        (y.new_zeros([0] * element_ndim) if (y.ndim == 1 and y.nelement() == 0) else y)
        for y in x
    ]

    if any(y.ndim != x[0].ndim for y in x):
        raise ValueError("All items have to have the same number of dimensions!")

    if pad_size is None:
        pad_dims = [
            max(y.shape[dim] for y in x if len(y) > 0) for dim in range(x[0].ndim)
        ]
    else:
        if any(len(pad_size) != y.ndim for y in x):
            raise ValueError("Pad size must contain target size for all dimensions.")
        pad_dims = pad_size

    N = len(x)
    x_padded = x[0].new_full((N, *pad_dims), pad_value)
    for i, y in enumerate(x):
        if len(y) > 0:
            slices = (i, *(slice(0, y.shape[dim]) for dim in range(y.ndim)))
            x_padded[slices] = y
    return x_padded


def wmean(
    x: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
    dim: Union[int, Tuple[int]] = -2,
    keepdim: bool = True,
    eps: float = 1e-9,
) -> torch.Tensor:
    """
    Finds the mean of the input tensor across the specified dimension.
    If the `weight` argument is provided, computes weighted mean.
    Args:
        x: tensor of shape `(*, D)`, where D is assumed to be spatial;
        weights: if given, non-negative tensor of shape `(*,)`. It must be
            broadcastable to `x.shape[:-1]`. Note that the weights for
            the last (spatial) dimension are assumed same;
        dim: dimension(s) in `x` to average over;
        keepdim: tells whether to keep the resulting singleton dimension.
        eps: minimum clamping value in the denominator.
    Returns:
        the mean tensor:
        * if `weights` is None => `mean(x, dim)`,
        * otherwise => `sum(x*w, dim) / max{sum(w, dim), eps}`.
    """
    args = {"dim": dim, "keepdim": keepdim}

    if weight is None:
        # pyre-fixme[6]: For 1st param expected `Optional[dtype]` but got
        #  `Union[Tuple[int], int]`.
        return x.mean(**args)

    if any(
        xd != wd and xd != 1 and wd != 1
        for xd, wd in zip(x.shape[-2::-1], weight.shape[::-1])
    ):
        raise ValueError("wmean: weights are not compatible with the tensor")

    # pyre-fixme[6]: For 1st param expected `Optional[dtype]` but got
    #  `Union[Tuple[int], int]`.
    return (x * weight[..., None]).sum(**args) / weight[..., None].sum(**args).clamp(
        eps
    )


def is_pointclouds(pcl) -> bool:
    """Checks whether the input `pcl` is an instance of `Pointclouds`
    by checking the existence of `points_padded` and `num_points_per_cloud`
    functions.
    """
    return hasattr(pcl, "points_padded") and hasattr(pcl, "num_points_per_cloud")


def eyes(
    dim: int,
    N: int,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Generates a batch of `N` identity matrices of shape `(N, dim, dim)`.
    Args:
        **dim**: The dimensionality of the identity matrices.
        **N**: The number of identity matrices.
        **device**: The device to be used for allocating the matrices.
        **dtype**: The datatype of the matrices.
    Returns:
        **identities**: A batch of identity matrices of shape `(N, dim, dim)`.
    """
    identities = torch.eye(dim, device=device, dtype=dtype)
    return identities[None].repeat(N, 1, 1)


def convert_pointclouds_to_tensor(pcl):
    """
    If `type(pcl)==Pointclouds`, converts a `pcl` object to a
    padded representation and returns it together with the number of points
    per batch. Otherwise, returns the input itself with the number of points
    set to the size of the second dimension of `pcl`.
    """
    if is_pointclouds(pcl):
        X = pcl.points_padded()  # type: ignore
        num_points = pcl.num_points_per_cloud()  # type: ignore
    elif torch.is_tensor(pcl):
        X = pcl
        num_points = X.shape[1] * torch.ones(  # type: ignore
            # pyre-fixme[16]: Item `Pointclouds` of `Union[Pointclouds, Tensor]` has
            #  no attribute `shape`.
            X.shape[0],
            device=X.device,
            dtype=torch.int64,
        )
    else:
        raise ValueError(
            "The inputs X, Y should be either Pointclouds objects or tensors."
        )
    return X, num_points

def corresponding_points_alignment(
    X,
    Y,
    weights: Union[torch.Tensor, List[torch.Tensor], None] = None,
    estimate_scale: bool = False,
    allow_reflection: bool = False,
    eps: float = 1e-9,
) -> SimilarityTransform:
    """
    Finds a similarity transformation (rotation `R`, translation `T`
    and optionally scale `s`)  between two given sets of corresponding
    `d`-dimensional points `X` and `Y` such that:
    `s[i] X[i] R[i] + T[i] = Y[i]`,
    for all batch indexes `i` in the least squares sense.
    The algorithm is also known as Umeyama [1].
    Args:
        **X**: Batch of `d`-dimensional points of shape `(minibatch, num_point, d)`
            or a `Pointclouds` object.
        **Y**: Batch of `d`-dimensional points of shape `(minibatch, num_point, d)`
            or a `Pointclouds` object.
        **weights**: Batch of non-negative weights of
            shape `(minibatch, num_point)` or list of `minibatch` 1-dimensional
            tensors that may have different shapes; in that case, the length of
            i-th tensor should be equal to the number of points in X_i and Y_i.
            Passing `None` means uniform weights.
        **estimate_scale**: If `True`, also estimates a scaling component `s`
            of the transformation. Otherwise assumes an identity
            scale and returns a tensor of ones.
        **allow_reflection**: If `True`, allows the algorithm to return `R`
            which is orthonormal but has determinant==-1.
        **eps**: A scalar for clamping to avoid dividing by zero. Active for the
            code that estimates the output scale `s`.
    Returns:
        3-element named tuple `SimilarityTransform` containing
        - **R**: Batch of orthonormal matrices of shape `(minibatch, d, d)`.
        - **T**: Batch of translations of shape `(minibatch, d)`.
        - **s**: batch of scaling factors of shape `(minibatch, )`.
    References:
        [1] Shinji Umeyama: Least-Suqares Estimation of
        Transformation Parameters Between Two Point Patterns
    """

    # make sure we convert input Pointclouds structures to tensors
    Xt, num_points = convert_pointclouds_to_tensor(X)
    Yt, num_points_Y = convert_pointclouds_to_tensor(Y)

    if (Xt.shape != Yt.shape) or (num_points != num_points_Y).any():
        raise ValueError(
            "Point sets X and Y have to have the same \
            number of batches, points and dimensions."
        )
    if weights is not None:
        if isinstance(weights, list):
            if any(np != w.shape[0] for np, w in zip(num_points, weights)):
                raise ValueError(
                    "number of weights should equal to the "
                    + "number of points in the point cloud."
                )
            weights = [w[..., None] for w in weights]
            weights = list_to_padded(weights)[..., 0]

        if Xt.shape[:2] != weights.shape:
            raise ValueError("weights should have the same first two dimensions as X.")

    b, n, dim = Xt.shape

    if (num_points < Xt.shape[1]).any() or (num_points < Yt.shape[1]).any():
        # in case we got Pointclouds as input, mask the unused entries in Xc, Yc
        mask = (
            torch.arange(n, dtype=torch.int64, device=Xt.device)[None]
            < num_points[:, None]
        ).type_as(Xt)
        weights = mask if weights is None else mask * weights.type_as(Xt)

    # compute the centroids of the point sets
    Xmu = wmean(Xt, weight=weights, eps=eps)
    Ymu = wmean(Yt, weight=weights, eps=eps)

    # mean-center the point sets
    Xc = Xt - Xmu
    Yc = Yt - Ymu

    total_weight = torch.clamp(num_points, 1)
    # special handling for heterogeneous point clouds and/or input weights
    if weights is not None:
        Xc *= weights[:, :, None]
        Yc *= weights[:, :, None]
        total_weight = torch.clamp(weights.sum(1), eps)

    if (num_points < (dim + 1)).any():
        warnings.warn(
            "The size of one of the point clouds is <= dim+1. "
            + "corresponding_points_alignment cannot return a unique rotation."
        )

    # compute the covariance XYcov between the point sets Xc, Yc
    XYcov = torch.bmm(Xc.transpose(2, 1), Yc)
    XYcov = XYcov / total_weight[:, None, None]

    # decompose the covariance matrix XYcov
    U, S, V = torch.svd(XYcov)

    # catch ambiguous rotation by checking the magnitude of singular values
    if (S.abs() <= AMBIGUOUS_ROT_SINGULAR_THR).any() and not (
        num_points < (dim + 1)
    ).any():
        warnings.warn(
            "Excessively low rank of "
            + "cross-correlation between aligned point clouds. "
            + "corresponding_points_alignment cannot return a unique rotation."
        )

    # identity matrix used for fixing reflections
    E = torch.eye(dim, dtype=XYcov.dtype, device=XYcov.device)[None].repeat(b, 1, 1)

    if not allow_reflection:
        # reflection test:
        #   checks whether the estimated rotation has det==1,
        #   if not, finds the nearest rotation s.t. det==1 by
        #   flipping the sign of the last singular vector U
        R_test = torch.bmm(U, V.transpose(2, 1))
        E[:, -1, -1] = torch.det(R_test)

    # find the rotation matrix by composing U and V again
    R = torch.bmm(torch.bmm(U, E), V.transpose(2, 1))

    if estimate_scale:
        # estimate the scaling component of the transformation
        trace_ES = (torch.diagonal(E, dim1=1, dim2=2) * S).sum(1)
        Xcov = (Xc * Xc).sum((1, 2)) / total_weight

        # the scaling component
        s = trace_ES / torch.clamp(Xcov, eps)

        # translation component
        T = Ymu[:, 0, :] - s[:, None] * torch.bmm(Xmu, R)[:, 0, :]
    else:
        # translation component
        T = Ymu[:, 0, :] - torch.bmm(Xmu, R)[:, 0, :]

        # unit scaling since we do not estimate scale
        s = T.new_ones(b)

    return SimilarityTransform(R, T, s)


def corresponding_cameras_alignment(
    cameras_src: "CamerasBase",
    cameras_tgt: "CamerasBase",
    estimate_scale: bool = True,
    mode: str = "extrinsics",
    eps: float = 1e-9,
) -> "CamerasBase":  # pragma: no cover
    """
    .. warning::
        The `corresponding_cameras_alignment` API is experimental
        and subject to change!
    Estimates a single similarity transformation between two sets of cameras
    `cameras_src` and `cameras_tgt` and returns an aligned version of
    `cameras_src`.
    Given source cameras [(R_1, T_1), (R_2, T_2), ..., (R_N, T_N)] and
    target cameras [(R_1', T_1'), (R_2', T_2'), ..., (R_N', T_N')],
    where (R_i, T_i) is a 2-tuple of the camera rotation and translation matrix
    respectively, the algorithm finds a global rotation, translation and scale
    (R_A, T_A, s_A) which aligns all source cameras with the target cameras
    such that the following holds:
        Under the change of coordinates using a similarity transform
        (R_A, T_A, s_A) a 3D point X' is mapped to X with: ::
            X = (X' R_A + T_A) / s_A
        Then, for all cameras `i`, we assume that the following holds: ::
            X R_i + T_i = s' (X' R_i' + T_i'),
        i.e. an adjusted point X' is mapped by a camera (R_i', T_i')
        to the same point as imaged from camera (R_i, T_i) after resolving
        the scale ambiguity with a global scalar factor s'.
        Substituting for X above gives rise to the following: ::
            (X' R_A + T_A) / s_A R_i + T_i = s' (X' R_i' + T_i')       // · s_A
            (X' R_A + T_A) R_i + T_i s_A = (s' s_A) (X' R_i' + T_i')
            s' := 1 / s_A  # without loss of generality
            (X' R_A + T_A) R_i + T_i s_A = X' R_i' + T_i'
            X' R_A R_i + T_A R_i + T_i s_A = X' R_i' + T_i'
               ^^^^^^^   ^^^^^^^^^^^^^^^^^
               ~= R_i'        ~= T_i'
        i.e. after estimating R_A, T_A, s_A, the aligned source cameras have
        extrinsics: ::
            cameras_src_align = (R_A R_i, T_A R_i + T_i s_A) ~= (R_i', T_i')
    We support two ways `R_A, T_A, s_A` can be estimated:
        1) `mode=='centers'`
            Estimates the similarity alignment between camera centers using
            Umeyama's algorithm (see `pytorch3d.ops.corresponding_points_alignment`
            for details) and transforms camera extrinsics accordingly.
        2) `mode=='extrinsics'`
            Defines the alignment problem as a system
            of the following equations: ::
                for all i:
                [ R_A   0 ] x [ R_i         0 ] = [ R_i' 0 ]
                [ T_A^T 1 ]   [ (s_A T_i^T) 1 ]   [ T_i' 1 ]
            `R_A, T_A` and `s_A` are then obtained by solving the
            system in the least squares sense.
    The estimated camera transformation is a true similarity transform, i.e.
    it cannot be a reflection.
    Args:
        cameras_src: `N` cameras to be aligned.
        cameras_tgt: `N` target cameras.
        estimate_scale: Controls whether the alignment transform is rigid
            (`estimate_scale=False`), or a similarity (`estimate_scale=True`).
            `s_A` is set to `1` if `estimate_scale==False`.
        mode: Controls the alignment algorithm.
            Can be one either `'centers'` or `'extrinsics'`. Please refer to the
            description above for details.
        eps: A scalar for clamping to avoid dividing by zero.
            Active when `estimate_scale==True`.
    Returns:
        cameras_src_aligned: `cameras_src` after applying the alignment transform.
    """

    if cameras_src.R.shape[0] != cameras_tgt.R.shape[0]:
        raise ValueError(
            "cameras_src and cameras_tgt have to contain the same number of cameras!"
        )

    if mode == "centers":
        align_fun = _align_camera_centers
    elif mode == "extrinsics":
        # raise NotImplementedError(" Only Umeyama's method has been implemented")
        align_fun = _align_camera_extrinsics
    else:
        raise ValueError("mode has to be one of (centers, extrinsics)")

    align_t_R, align_t_T, align_t_s = align_fun(
        cameras_src, cameras_tgt, estimate_scale=estimate_scale, eps=eps
    )

    # create a new cameras object and set the R and T accordingly
    cameras_src_aligned = cameras_src.clone()
    # pyre-fixme[6]: For 2nd param expected `Tensor` but got `Union[Tensor, Module]`.
    cameras_src_aligned.R = torch.bmm(align_t_R.expand_as(cameras_src.R), cameras_src.R)
    cameras_src_aligned.T = (
        torch.bmm(
            align_t_T[:, None].repeat(cameras_src.R.shape[0], 1, 1),
            # pyre-fixme[6]: For 2nd param expected `Tensor` but got `Union[Tensor,
            #  Module]`.
            cameras_src.R,
        )[:, 0]
        # pyre-fixme[29]: `Union[BoundMethod[typing.Callable(torch._C._TensorBase.__m...
        + cameras_src.T * align_t_s
    )

    return cameras_src_aligned


def _align_camera_centers(
    cameras_src,
    cameras_tgt,
    estimate_scale: bool = True,
    eps: float = 1e-9,
):  # pragma: no cover
    """
    Use Umeyama's algorithm to align the camera centers.
    """
    centers_src = cameras_src # torch.tensor (n,3)
    centers_tgt = cameras_tgt # torch.tensor (n,3)
    align_t = corresponding_points_alignment(
        centers_src[None],
        centers_tgt[None],
        estimate_scale=estimate_scale,
        allow_reflection=False,
        eps=eps,
    )
    # the camera transform is the inverse of the estimated transform between centers
    # align_t_R = align_t.R.permute(0, 2, 1)
    # align_t_T = -(torch.bmm(align_t.T[:, None], align_t_R))[:, 0]
    # align_t_s = align_t.s[0]

    align_t_R = align_t.R
    align_t_T = align_t.T
    align_t_s = align_t.s[0]

    

    return align_t_R, align_t_T, align_t_s


def _align_camera_extrinsics(
    cameras_src,
    cameras_tgt,
    estimate_scale: bool = True,
    eps: float = 1e-9,
):  # pragma: no cover
    """
    Get the global rotation R_A with svd of cov(RR^T):
        ```
        R_A R_i = R_i' for all i
        R_A [R_1 R_2 ... R_N] = [R_1' R_2' ... R_N']
        U, _, V = svd([R_1 R_2 ... R_N]^T [R_1' R_2' ... R_N'])
        R_A = (U V^T)^T
        ```
    """
    # camera_src_R = 
    RRcov = torch.bmm(cameras_src.R, cameras_tgt.R.transpose(2, 1)).mean(0)
    U, _, V = torch.svd(RRcov)
    align_t_R = V @ U.t()

    """
    The translation + scale `T_A` and `s_A` is computed by finding
    a translation and scaling that aligns two tensors `A, B`
    defined as follows:
        ```
        T_A R_i + s_A T_i   = T_i'        ;  for all i    // · R_i^T
        s_A T_i R_i^T + T_A = T_i' R_i^T  ;  for all i
            ^^^^^^^^^         ^^^^^^^^^^
                A_i                B_i

        A_i := T_i R_i^T
        A = [A_1 A_2 ... A_N]
        B_i := T_i' R_i^T
        B = [B_1 B_2 ... B_N]
        ```
    The scale s_A can be retrieved by matching the correlations of
    the points sets A and B:
        ```
        s_A = (A-mean(A))*(B-mean(B)).sum() / ((A-mean(A))**2).sum()
        ```
    The translation `T_A` is then defined as:
        ```
        T_A = mean(B) - mean(A) * s_A
        ```
    """
    A = torch.bmm(cameras_src.R, cameras_src.T[:, :, None])[:, :, 0]
    B = torch.bmm(cameras_src.R, cameras_tgt.T[:, :, None])[:, :, 0]
    Amu = A.mean(0, keepdim=True)
    Bmu = B.mean(0, keepdim=True)
    if estimate_scale and A.shape[0] > 1:
        # get the scaling component by matching covariances
        # of centered A and centered B
        Ac = A - Amu
        Bc = B - Bmu
        # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and `int`.
        align_t_s = (Ac * Bc).mean() / (Ac**2).mean().clamp(eps)
    else:
        # set the scale to identity
        align_t_s = 1.0
    # get the translation as the difference between the means of A and B
    align_t_T = Bmu - align_t_s * Amu

    return align_t_R, align_t_T, align_t_s

def _apply_similarity_transform(
    X: torch.Tensor, R: torch.Tensor, T: torch.Tensor, s: torch.Tensor
) -> torch.Tensor:
    """
    Applies a similarity transformation parametrized with a batch of orthonormal
    matrices `R` of shape `(minibatch, d, d)`, a batch of translations `T`
    of shape `(minibatch, d)` and a batch of scaling factors `s`
    of shape `(minibatch,)` to a given `d`-dimensional cloud `X`
    of shape `(minibatch, num_points, d)`
    """
    X = s[:, None, None] * torch.bmm(X, R) + T[:, None, :]
    return X


