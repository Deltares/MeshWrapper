import numpy as np
import pytest
from pytest import approx

from meshkernel import (
    CurvilinearParameters,
    MakeGridParameters,
    GeometryList,
    MeshKernel,
    SplinesToCurvilinearParameters,
    OrthogonalizationParameters
)


def test_curvilinear_compute_transfinite_from_splines():
    r"""Tests `curvilinear_compute_transfinite_from_splines` generates a curvilinear grid.
    """
    mk = MeshKernel()

    separator = -999.0
    splines_x = np.array([2.172341E+02, 4.314185E+02, 8.064374E+02, separator,
                          2.894012E+01, 2.344944E+02, 6.424647E+02, separator,
                          2.265137E+00, 2.799988E+02, separator,
                          5.067361E+02, 7.475956E+02], dtype=np.double)
    splines_y = np.array([-2.415445E+01, 1.947381E+02, 3.987241E+02, separator,
                          2.010146E+02, 3.720490E+02, 5.917262E+02, separator,
                          2.802553E+02, -2.807726E+01, separator,
                          6.034946E+02, 3.336055E+02], dtype=np.double)
    splines_values = np.zeros_like(splines_x)
    splines = GeometryList(splines_x, splines_y, splines_values)

    curvilinearParameters = CurvilinearParameters()
    curvilinearParameters.n_refinement = 40
    curvilinearParameters.m_refinement = 20

    mk.curvilinear_compute_transfinite_from_splines(splines, curvilinearParameters)

    output_curvilinear = mk.curvilineargrid_get()

    # Test the number of m and n are as expected
    assert output_curvilinear.num_m == 21
    assert output_curvilinear.num_n == 41


def test_curvilinear_compute_orthogonal_from_splines():
    r"""Tests `curvilinear_compute_orthogonal_from_splines` generates a curvilinear grid.
    """
    mk = MeshKernel()

    separator = -999.0
    splines_x = np.array([152.001571655273, 374.752960205078, 850.255920410156, separator,
                          72.5010681152344, 462.503479003906, separator], dtype=np.double)
    splines_y = np.array([86.6264953613281, 336.378997802734, 499.130676269531, separator,
                          391.129577636719, 90.3765411376953, separator], dtype=np.double)

    splines_values = np.zeros_like(splines_x)
    splines = GeometryList(splines_x, splines_y, splines_values)

    curvilinear_parameters = CurvilinearParameters()
    curvilinear_parameters.n_refinement = 40
    curvilinear_parameters.m_refinement = 20

    splines_to_curvilinear_parameters = SplinesToCurvilinearParameters()
    splines_to_curvilinear_parameters.aspect_ratio = 0.1
    splines_to_curvilinear_parameters.aspect_ratio_grow_factor = 1.1
    splines_to_curvilinear_parameters.average_width = 500.0
    splines_to_curvilinear_parameters.nodes_on_top_of_each_other_tolerance = 1e-4
    splines_to_curvilinear_parameters.min_cosine_crossing_angles = 0.95
    splines_to_curvilinear_parameters.check_front_collisions = 0
    splines_to_curvilinear_parameters.curvature_adapted_grid_spacing = 1
    splines_to_curvilinear_parameters.remove_skinny_triangles = 0

    mk.curvilinear_compute_orthogonal_from_splines(splines, curvilinear_parameters, splines_to_curvilinear_parameters)

    output_curvilinear = mk.curvilineargrid_get()

    # Test the number of m and n are as expected
    assert output_curvilinear.num_m == 3
    assert output_curvilinear.num_n == 9


def test_curvilinear_compute_transfinite_from_splines():
    r"""Tests `curvilinear_compute_transfinite_from_splines` converts a curvilinear mesh into an unstructured mesh.
    """
    mk = MeshKernel()

    separator = -999.0
    splines_x = np.array([2.0, 4.0, 7.0, separator,
                          -1.0, 1.0, 5.0, separator,
                          3.0, -2.0, separator,
                          7.0, 4.0], dtype=np.double)
    splines_y = np.array([1.0, 3.0, 4.0, separator,
                          4.0, 6.0, 7.0, separator,
                          1.0, 6.0, separator,
                          3.0, 8.0], dtype=np.double)
    splines_values = np.zeros_like(splines_x)
    splines = GeometryList(splines_x, splines_y, splines_values)

    curvilinear_parameters = CurvilinearParameters()
    curvilinear_parameters.n_refinement = 10
    curvilinear_parameters.m_refinement = 10

    mk.curvilinear_compute_transfinite_from_splines(splines, curvilinear_parameters)

    mk.curvilinear_convert_to_mesh2d()

    mesh2d = mk.mesh2d_get()

    curvilinear_grid = mk.curvilineargrid_get()

    # Test the number of m and n are as expected
    assert curvilinear_grid.num_m == 0
    assert curvilinear_grid.num_n == 0
    assert len(mesh2d.node_x) == 121
    assert len(mesh2d.edge_nodes) == 440


def test_curvilinear_refine():
    r"""Tests `curvilinear_refine` refines a curvilinear grid in a defined block.
    """
    mk = MeshKernel()

    make_grid_parameters = MakeGridParameters()
    make_grid_parameters.num_columns = 3
    make_grid_parameters.num_rows = 3
    make_grid_parameters.angle = 0.0
    make_grid_parameters.block_size = 0.0
    make_grid_parameters.origin_x = 0.0
    make_grid_parameters.origin_y = 0.0
    make_grid_parameters.block_size_x = 10.0
    make_grid_parameters.block_size_y = 10.0

    node_x = np.empty(0, dtype=np.double)
    node_y = np.empty(0, dtype=np.double)
    geometry_list = GeometryList(node_x, node_y)

    mk.curvilinear_make_uniform(make_grid_parameters, geometry_list)

    curvilinear_grid = mk.curvilineargrid_get()

    # Test the number of m and n before refinement
    assert curvilinear_grid.num_m == 4
    assert curvilinear_grid.num_n == 4

    mk.curvilinear_refine(10.0, 20.0, 20.0, 20.0, 10)

    curvilinear_grid = mk.curvilineargrid_get()

    # Test the number of m and n after refinement
    assert curvilinear_grid.num_m == 4
    assert curvilinear_grid.num_n == 13


def test_curvilinear_derefine():
    r"""Tests `curvilinear_derefine` de-refines a curvilinear grid .
    """
    mk = MeshKernel()

    make_grid_parameters = MakeGridParameters()
    make_grid_parameters.num_columns = 10
    make_grid_parameters.num_rows = 10
    make_grid_parameters.angle = 0.0
    make_grid_parameters.block_size = 0.0
    make_grid_parameters.origin_x = 0.0
    make_grid_parameters.origin_y = 0.0
    make_grid_parameters.block_size_x = 10.0
    make_grid_parameters.block_size_y = 10.0

    node_x = np.empty(0, dtype=np.double)
    node_y = np.empty(0, dtype=np.double)
    geometry_list = GeometryList(node_x, node_y)

    mk.curvilinear_make_uniform(make_grid_parameters, geometry_list)

    # First perform refinement
    mk.curvilinear_refine(10.0, 20.0, 20.0, 20.0, 10)

    curvilinear_grid = mk.curvilineargrid_get()

    # Test the number of n increased
    assert curvilinear_grid.num_n == 20

    mk.curvilinear_derefine(10.0, 20.0, 20.0, 20.0)

    curvilinear_grid = mk.curvilineargrid_get()

    # Test the number of n decreased to its original value
    assert curvilinear_grid.num_n == 11


def test_curvilinear_compute_transfinite_from_polygon():
    r"""Tests `curvilinear_compute_transfinite_from_polygon` generates curvilinear grid .

    Input polygon:
    6---5---4
    |       |
    7       3
    |       |
    0---1---2

    Generated curvilineargrid:

    6---7---8
    |   |   |
    3---4---5
    |   |   |
    0---1---2
    """
    node_x = np.array([0, 5, 10, 10, 10, 5, 0, 0, 0], dtype=np.double)
    node_y = np.array([0, 0, 0, 5, 10, 10, 10, 5, 0], dtype=np.double)
    geometry_list = GeometryList(node_x, node_y)

    mk = MeshKernel()

    mk.curvilinear_compute_transfinite_from_polygon(geometry_list, 0, 2, 4, False)

    curvilinear_grid = mk.curvilineargrid_get()

    # Test ta curvilinear grid was generated
    assert curvilinear_grid.num_m == 3
    assert curvilinear_grid.num_n == 3


def test_curvilinear_compute_transfinite_from_triangle():
    r"""Tests `curvilinear_compute_transfinite_from_triangle` computes a curvilinear grid.
    """

    node_x = np.array([444.504791,
                       427.731781,
                       405.640503,
                       381.094666,
                       451.050354,
                       528.778931,
                       593.416260,
                       558.643005,
                       526.733398,
                       444.095703], dtype=np.double)

    node_y = np.array([437.155945,
                       382.745758,
                       317.699005,
                       262.470612,
                       262.879700,
                       263.288788,
                       266.561584,
                       324.653687,
                       377.836578,
                       436.746857], dtype=np.double)

    geometry_list = GeometryList(node_x, node_y)

    mk = MeshKernel()

    mk.curvilinear_compute_transfinite_from_triangle(geometry_list, 0, 3, 6)

    curvilinear_grid = mk.curvilineargrid_get()

    # Test ta curvilinear grid was generated
    assert curvilinear_grid.num_m == 4
    assert curvilinear_grid.num_n == 4


def test_curvilinear_grid_orthogonalization():
    r"""Tests `curvilinear_initialize_orthogonalize` computes a curvilinear grid.
    """

    # Create a new curvilinear grid instance
    mk = MeshKernel()

    separator = -999.0
    splines_x = np.array([2.0, 4.0, 7.0, separator,
                          -1.0, 1.0, 5.0, separator,
                          3.0, -2.0, separator,
                          7.0, 4.0], dtype=np.double)
    splines_y = np.array([1.0, 3.0, 4.0, separator,
                          4.0, 6.0, 7.0, separator,
                          1.0, 6.0, separator,
                          3.0, 8.0], dtype=np.double)
    splines_values = np.zeros_like(splines_x)
    splines = GeometryList(splines_x, splines_y, splines_values)

    curvilinear_parameters = CurvilinearParameters()
    curvilinear_parameters.n_refinement = 10
    curvilinear_parameters.m_refinement = 10

    mk.curvilinear_compute_transfinite_from_splines(splines, curvilinear_parameters)

    # Assert a nodal position before orthogonalization
    curvilinear_grid = mk.curvilineargrid_get()
    assert curvilinear_grid.node_x[1] == approx(2.1380641421159616, 0.0001)
    assert curvilinear_grid.node_y[1] == approx(1.861935857884038, 0.0001)

    orthogonalization_parameters = OrthogonalizationParameters()
    orthogonalization_parameters.outer_iterations = 1
    orthogonalization_parameters.boundary_iterations = 25
    orthogonalization_parameters.inner_iterations = 25
    orthogonalization_parameters.orthogonalization_to_smoothing_factor = 0.975

    # Initialize the curvilinear grid orthogonalization algorithm
    mk.curvilinear_initialize_orthogonalize(orthogonalization_parameters)

    # Initialize the curvilinear grid orthogonalization algorithm
    mk.curvilinear_set_block_orthogonalize(2.43, 1.56, 4.63, 6.93)

    # Performs orthogonalization
    mk.curvilinear_orthogonalize()

    curvilinear_grid = mk.curvilineargrid_get()

    # Assert the nodal position after orthogonalization
    assert curvilinear_grid.node_x[1] == approx(2.1656235953439653, 0.0001)
    assert curvilinear_grid.node_y[1] == approx(1.8343764046560345, 0.0001)
