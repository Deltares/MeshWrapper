import numpy as np
import pytest
from numpy.ctypeslib import as_array
from numpy.testing import assert_array_equal

from meshkernel import (
    GeometryList,
    InterpolationParameters,
    Mesh2d,
    OrthogonalizationParameters,
    SampleRefineParameters,
)
from meshkernel.c_structures import (
    CGeometryList,
    CInterpolationParameters,
    CMesh2d,
    COrthogonalizationParameters,
    CSampleRefineParameters,
)


def test_cmesh2d_from_mesh2d():
    """Tests `from_mesh2d` of the `CMesh2D` class with a simple mesh."""

    # 2---3
    # |   |
    # 0---1
    node_x = np.array([0.0, 1.0, 1.0, 0.0], dtype=np.double)
    node_y = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.double)
    edge_nodes = np.array([0, 1, 1, 3, 3, 2, 2, 0], dtype=np.int32)
    face_nodes = np.array([0, 1, 2, 3], dtype=np.int32)
    nodes_per_face = np.array([4], dtype=np.int32)
    edge_x = np.array([0.5, 1.0, 0.5, 0.0], dtype=np.double)
    edge_y = np.array([0.0, 0.5, 1.0, 0.5], dtype=np.double)
    face_x = np.array([0.5], dtype=np.double)
    face_y = np.array([0.5], dtype=np.double)

    mesh2d = Mesh2d(node_x, node_y, edge_nodes)
    mesh2d.face_nodes = face_nodes
    mesh2d.nodes_per_face = nodes_per_face
    mesh2d.edge_x = edge_x
    mesh2d.edge_y = edge_y
    mesh2d.face_x = face_x
    mesh2d.face_y = face_y

    c_mesh2d = CMesh2d.from_mesh2d(mesh2d)

    # Get the numpy arrays from the ctypes object
    c_mesh2d_node_x = as_array(c_mesh2d.node_x, (4,))
    c_mesh2d_node_y = as_array(c_mesh2d.node_y, (4,))
    c_mesh2d_edge_nodes = as_array(c_mesh2d.edge_nodes, (8,))
    c_mesh2d_face_nodes = as_array(c_mesh2d.face_nodes, (4,))
    c_mesh2d_nodes_per_face = as_array(c_mesh2d.nodes_per_face, (1,))
    c_mesh2d_edge_x = as_array(c_mesh2d.edge_x, (4,))
    c_mesh2d_edge_y = as_array(c_mesh2d.edge_y, (4,))
    c_mesh2d_face_x = as_array(c_mesh2d.face_x, (1,))
    c_mesh2d_face_y = as_array(c_mesh2d.face_y, (1,))

    # Assert data is correct
    assert_array_equal(c_mesh2d_node_x, node_x)
    assert_array_equal(c_mesh2d_node_y, node_y)
    assert_array_equal(c_mesh2d_edge_nodes, edge_nodes)
    assert_array_equal(c_mesh2d_face_nodes, face_nodes)
    assert_array_equal(c_mesh2d_nodes_per_face, nodes_per_face)
    assert_array_equal(c_mesh2d_edge_x, edge_x)
    assert_array_equal(c_mesh2d_edge_y, edge_y)
    assert_array_equal(c_mesh2d_face_x, face_x)
    assert_array_equal(c_mesh2d_face_y, face_y)

    assert c_mesh2d.num_nodes == 4
    assert c_mesh2d.num_edges == 4
    assert c_mesh2d.num_faces == 1
    assert c_mesh2d.num_face_nodes == 4


def test_cmesh2d_allocate_memory():
    """Tests `allocate_memory` of the `CMesh2D` class."""

    c_mesh2d = CMesh2d()
    c_mesh2d.num_nodes = 4
    c_mesh2d.num_edges = 4
    c_mesh2d.num_faces = 1
    c_mesh2d.num_face_nodes = 4

    mesh2d = c_mesh2d.allocate_memory()

    assert mesh2d.node_x.size == 4
    assert mesh2d.node_y.size == 4
    assert mesh2d.edge_nodes.size == 8
    assert mesh2d.face_nodes.size == 4
    assert mesh2d.nodes_per_face.size == 1
    assert mesh2d.edge_x.size == 4
    assert mesh2d.edge_y.size == 4
    assert mesh2d.face_x.size == 1
    assert mesh2d.face_y.size == 1


def test_cgeometrylist_from_geometrylist():
    """Tests `from_geometrylist` of the `CGeometryList` class."""

    x_coordinates = np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.double)
    y_coordinates = np.array([5.0, 6.0, 7.0, 8.0, 9.0], dtype=np.double)
    values = np.array([10.0, 11.0, 12.0, 13.0, 14.0], dtype=np.double)
    geometry_separator = 15.0
    inner_outer_separator = 16.0

    geometry_list = GeometryList(x_coordinates, y_coordinates)
    geometry_list.values = values
    geometry_list.geometry_separator = geometry_separator
    geometry_list.inner_outer_separator = inner_outer_separator

    c_geometry_list = CGeometryList.from_geometrylist(geometry_list)

    # Get the numpy arrays from the ctypes object
    c_geometry_list_x_coordinates = as_array(c_geometry_list.x_coordinates, (5,))
    c_geometry_list_y_coordinates = as_array(c_geometry_list.y_coordinates, (5,))
    c_geometry_list_values = as_array(c_geometry_list.values, (5,))

    assert_array_equal(c_geometry_list_x_coordinates, x_coordinates)
    assert_array_equal(c_geometry_list_y_coordinates, y_coordinates)
    assert_array_equal(c_geometry_list_values, values)

    assert c_geometry_list.geometry_separator == geometry_separator
    assert c_geometry_list.inner_outer_separator == inner_outer_separator
    assert c_geometry_list.n_coordinates == x_coordinates.size


def test_corthogonalizationparameters_from_orthogonalizationparameters():
    """Tests `from_orthogonalizationparameters` of the `COrthogonalizationParameters` class."""

    parameters = OrthogonalizationParameters()
    parameters.outer_iterations = 0
    parameters.boundary_iterations = 1
    parameters.inner_iterations = 2
    parameters.orthogonalization_to_smoothing_factor = 3.0
    parameters.orthogonalization_to_smoothing_factor_at_boundary = 4.0
    parameters.areal_to_angle_smoothing_factor = 5.0

    c_parameters = COrthogonalizationParameters.from_orthogonalizationparameters(
        parameters
    )

    assert c_parameters.outer_iterations == 0
    assert c_parameters.boundary_iterations == 1
    assert c_parameters.inner_iterations == 2
    assert c_parameters.orthogonalization_to_smoothing_factor == 3.0
    assert c_parameters.orthogonalization_to_smoothing_factor_at_boundary == 4.0
    assert c_parameters.areal_to_angle_smoothing_factor == 5.0


def test_cinterpolationparameters_from_interpolationparameters():
    """Tests `from_interpolationparameters` of the `CInterpolationParameters` class."""

    parameters = InterpolationParameters(0, 1)
    parameters.max_refinement_iterations = 2
    parameters.averaging_method = 3
    parameters.min_points = 4
    parameters.relative_search_radius = 5.0
    parameters.interpolate_to = 6

    c_parameters = CInterpolationParameters.from_interpolationparameters(parameters)

    assert c_parameters.refine_intersected == 0
    assert c_parameters.use_mass_center_when_refining == 1
    assert c_parameters.max_refinement_iterations == 2
    assert c_parameters.averaging_method == 3
    assert c_parameters.min_points == 4
    assert c_parameters.relative_search_radius == 5.0
    assert c_parameters.interpolate_to == 6


def test_csamplerefineparameters_from_samplerefinementparameters():
    """Tests `from_samplerefinementparameters` of the `CSampleRefineParameters` class."""

    parameters = SampleRefineParameters(4, 3.0, 2, True, 5, False)

    c_parameters = CSampleRefineParameters.from_samplerefinementparameters(parameters)

    assert c_parameters.max_refinement_iterations == 4
    assert c_parameters.min_face_size == 3.0
    assert c_parameters.refinement_type == 2
    assert c_parameters.connect_hanging_nodes == 1
    assert c_parameters.max_time_step == 5
    assert c_parameters.account_for_samples_outside_face == 0
