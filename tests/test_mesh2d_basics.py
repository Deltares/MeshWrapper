import numpy as np
import pytest
from numpy import ndarray
from numpy.testing import assert_array_almost_equal, assert_array_equal
from pytest import approx
from transformation_utils import rotate, translate

from meshkernel import (
    DeleteMeshOption,
    GeometryList,
    GriddedSamples,
    InputError,
    InterpolationValues,
    MakeGridParameters,
    Mesh2d,
    Mesh2dLocation,
    MeshKernel,
    MeshKernelError,
    MeshRefinementParameters,
    ProjectionType,
    RefinementType,
)

cases_projection_constructor = [(ProjectionType.CARTESIAN), (ProjectionType.SPHERICAL)]


@pytest.mark.parametrize("projection", cases_projection_constructor)
def test_constructor(projection: ProjectionType):
    """Test if the constructor works"""
    MeshKernel(projection)


def test_different_instances_have_different_ids():
    """Test if the meshkernelid of two instances differs"""
    mk_1 = MeshKernel()
    mk_2 = MeshKernel()

    assert mk_1._meshkernelid != mk_2._meshkernelid


def test_mesh2d_set_and_mesh2d_get():
    """Test to set a simple mesh and then get it again with new parameters

    3---2
    |   |
    0---1
    """
    mk = MeshKernel()

    edge_nodes = np.array([0, 1, 1, 2, 2, 3, 3, 0], dtype=np.int32)
    node_x = np.array([0.0, 1.0, 1.0, 0.0], dtype=np.double)
    node_y = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.double)

    input_mesh2d = Mesh2d(node_x, node_y, edge_nodes)
    mk.mesh2d_set(input_mesh2d)

    output_mesh2d = mk.mesh2d_get()

    # Test if the input and output differs
    assert_array_equal(output_mesh2d.edge_nodes, input_mesh2d.edge_nodes)
    assert_array_equal(output_mesh2d.node_x, input_mesh2d.node_x)
    assert_array_equal(output_mesh2d.node_y, input_mesh2d.node_y)

    # Test if faces are correctly calculated
    assert_array_equal(output_mesh2d.face_nodes, np.array([0, 1, 2, 3]))
    assert_array_equal(output_mesh2d.nodes_per_face, np.array([4]))
    assert_array_equal(output_mesh2d.face_x, np.array([0.5]))
    assert_array_equal(output_mesh2d.face_y, np.array([0.5]))

    # Test if edges are correctly calculated
    assert_array_equal(output_mesh2d.edge_x, np.array([0.5, 1.0, 0.5, 0.0]))
    assert_array_equal(output_mesh2d.edge_y, np.array([0.0, 0.5, 1.0, 0.5]))


def test_mesh2d_add():
    """Test adding a 2d mesh"""
    mk = MeshKernel()

    edge_nodes = np.array([0, 1, 1, 2, 2, 3, 3, 0], dtype=np.int32)
    node_x = np.array([0.0, 1.0, 1.0, 0.0], dtype=np.double)
    node_y = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.double)

    input_mesh2d_1 = Mesh2d(node_x, node_y, edge_nodes)
    mk.mesh2d_set(input_mesh2d_1)

    input_mesh2d_2 = Mesh2d(node_x + 1, node_y, edge_nodes)
    mk.mesh2d_add(input_mesh2d_2)

    output_mesh2d = mk.mesh2d_get()

    assert_array_equal(
        output_mesh2d.node_x,
        np.concatenate(
            (input_mesh2d_1.node_x, input_mesh2d_2.node_x),
            axis=None,
        ),
    )

    assert_array_equal(
        output_mesh2d.node_y,
        np.concatenate(
            (input_mesh2d_1.node_y, input_mesh2d_2.node_y),
            axis=None,
        ),
    )


def test_mesh2d_insert_edge(meshkernel_with_mesh2d: MeshKernel):
    """Test `mesh2d_insert_edge` by inserting one edge within a 1x1 Mesh2d.

    2---3
    |   |
    0---1
    """

    mk = meshkernel_with_mesh2d(1, 1)

    edge_index = mk.mesh2d_insert_edge(0, 3)

    mesh2d = mk.mesh2d_get()

    assert edge_index == 4
    assert mesh2d.node_x.size == 4
    assert mesh2d.edge_x.size == 5
    assert mesh2d.face_x.size == 2


def test_mesh2d_insert_node(meshkernel_with_mesh2d: MeshKernel):
    """Test `mesh2d_insert_node` with a 1x1 Mesh2d.

    2---3
    |   |
    0---1
    """

    mk = meshkernel_with_mesh2d(1, 1)

    node_index = mk.mesh2d_insert_node(1.5, 0.5)
    edge_index = mk.mesh2d_insert_edge(3, node_index)

    mesh2d = mk.mesh2d_get()

    assert node_index == 4
    assert mesh2d.node_x.size == 5
    assert edge_index == 4
    assert mesh2d.edge_x.size == 5


cases_mesh2d_delete_node = [
    (0, 0.0, 0.0),
    (1, 1.0, 0.0),
    (2, 2.0, 0.0),
    (3, 0.0, 1.0),
    (4, 1.0, 1.0),
    (5, 2.0, 1.0),
    (6, 0.0, 2.0),
    (7, 1.0, 2.0),
    (8, 2.0, 2.0),
]


@pytest.mark.parametrize("node_index, deleted_x, deleted_y", cases_mesh2d_delete_node)
def test_mesh2d_delete_node(
    meshkernel_with_mesh2d: MeshKernel,
    node_index: int,
    deleted_x: float,
    deleted_y: float,
):
    """Test `mesh2d_delete_node` by deleting a node from a 2x2 Mesh2d.

    6---7---8
    |   |   |
    3---4---5
    |   |   |
    0---1---2

    """
    mk = meshkernel_with_mesh2d(2, 2)

    mk.mesh2d_delete_node(node_index)

    mesh2d = mk.mesh2d_get()

    assert mesh2d.node_x.size == 8

    for x, y in zip(mesh2d.node_x, mesh2d.node_y):
        assert x != deleted_x or y != deleted_y


def test_mesh2d_delete_node_invalid_node_index(meshkernel_with_mesh2d: MeshKernel):
    """Test `mesh2d_delete_node` by passing a negative `node_index`."""

    mk = meshkernel_with_mesh2d(1, 1)

    with pytest.raises(InputError):
        mk.mesh2d_delete_node(-1)


cases_mesh2d_move_node = [
    (0, 0.0, 0.0),
    (1, 1.0, 0.0),
    (2, 2.0, 0.0),
    (3, 0.0, 1.0),
    (4, 1.0, 1.0),
    (5, 2.0, 1.0),
    (6, 0.0, 2.0),
    (7, 1.0, 2.0),
    (8, 2.0, 2.0),
]


@pytest.mark.parametrize("node_index, moved_x, moved_y", cases_mesh2d_move_node)
def test_mesh2d_move_node(
    meshkernel_with_mesh2d: MeshKernel, node_index: int, moved_x: float, moved_y: float
):
    """Test to move a node in a simple Mesh2d to new location.

    6---7---8
    |   |   |
    3---4---5
    |   |   |
    0---1---2

    """

    mk = meshkernel_with_mesh2d(2, 2)

    mk.mesh2d_move_node(5.0, 7.0, node_index)

    mesh2d = mk.mesh2d_get()

    assert mesh2d.node_x[node_index] == 5.0
    assert mesh2d.node_y[node_index] == 7.0

    for x, y in zip(mesh2d.node_x, mesh2d.node_y):
        assert x != moved_x or y != moved_y


def test_mesh2d_move_node_invalid_node_index(meshkernel_with_mesh2d: MeshKernel):
    """Test `mesh2d_move_node` by passing a negative `node_index`."""

    mk = meshkernel_with_mesh2d(1, 1)
    with pytest.raises(InputError):
        mk.mesh2d_move_node(5.0, 7.0, -1)


cases_mesh2d_delete_edge = [
    (0.5, 0.0),
    (1.5, 0.0),
    (0.0, 0.5),
    (1.0, 0.5),
    (2.0, 0.5),
    (0.5, 1.0),
    (1.5, 1.0),
    (0.0, 1.5),
    (1.0, 1.5),
    (2.0, 1.5),
    (0.5, 2.0),
    (1.5, 2.0),
]


@pytest.mark.parametrize("delete_x, delete_y", cases_mesh2d_delete_edge)
def test_mesh2d_delete_edge(
    meshkernel_with_mesh2d: MeshKernel, delete_x: float, delete_y: float
):
    """Test `mesh2d_delete_edge` by deleting an edge from a 2x2 Mesh2d.

    6---7---8
    |   |   |
    3---4---5
    |   |   |
    0---1---2

    """
    mk = meshkernel_with_mesh2d(2, 2)

    mk.mesh2d_delete_edge(delete_x, delete_y)

    mesh2d = mk.mesh2d_get()

    assert mesh2d.node_x.size == 9
    assert mesh2d.edge_x.size == 11
    assert mesh2d.face_x.size == 3

    for x, y in zip(mesh2d.edge_x, mesh2d.edge_y):
        assert x != delete_x or y != delete_y


cases_mesh2d_get_edge = [
    (0.5, 0.0, 2),
    (1.0, 0.5, 1),
    (0.5, 1.0, 3),
    (0.0, 0.5, 0),
]


@pytest.mark.parametrize("x, y, exp_index", cases_mesh2d_get_edge)
def test_mesh2d_get_edge(
    meshkernel_with_mesh2d: MeshKernel, x: float, y: float, exp_index: int
):
    """Test `mesh2d_get_edge` on a 2x2 Mesh2d.

        (3)
       2---3
    (0)|   |(1)
       0---1
        (2)

    """

    mk = meshkernel_with_mesh2d(1, 1)

    edge_index = mk.mesh2d_get_edge(x, y)

    assert edge_index == exp_index


cases_mesh2d_get_node_index = [
    (0.0, 0.0, 0),
    (0.4, 0.0, 0),
    (0.0, 0.4, 0),
    (1.0, 0.0, 1),
    (0.6, 0.0, 1),
    (1.0, 0.4, 1),
    (0.0, 1.0, 2),
    (0.4, 1.0, 2),
    (0.0, 0.6, 2),
    (1.0, 1.0, 3),
    (0.6, 1.0, 3),
    (1.0, 0.6, 3),
]


@pytest.mark.parametrize("x, y, exp_index", cases_mesh2d_get_node_index)
def test_mesh2d_get_node_index(
    meshkernel_with_mesh2d: MeshKernel, x: float, y: float, exp_index: int
):
    """Test `mesh2d_get_node_index` on a 1x1 Mesh2d.

    2---3
    |   |
    0---1

    """

    mk = meshkernel_with_mesh2d(1, 1)

    edge_index = mk.mesh2d_get_node_index(x, y, 0.5)

    assert edge_index == exp_index


def test_mesh2d_get_node_index_no_node_in_search_radius(
    meshkernel_with_mesh2d: MeshKernel,
):
    """Test `get_node_index` when there is no node within the search radius."""

    mk = meshkernel_with_mesh2d(1, 1)

    index_value = mk.mesh2d_get_node_index(0.5, 0.5, 0.4)

    assert index_value == -1


# Case 1: should keep the central cell
#  20--21
#  |   |
#  14--15
# nodes = 4
# edges = 1
# faces = 1

# Case 2: should keep 3x3 central cells
#  25--26--27--28
#  |   |   |   |
#  19--20--21--22
#  |   |   |   |
#  13--14--15--16
#  |   |   |   |
#  7---8---9---10
# nodes = 4 * 4 = 16
# edges = 2 * (3 * 4) = 24
# faces = 1

# Case 3: should keep all cells but the central cell, i.e. delete the central face
#  30--31--32--33--34--35
#  |   |   |   |   |   |
#  24--25--26--27--28--29
#  |   |   |   |   |   |
#  18--19--20--21--22--23
#  |   |   | / |   |   |
#  12--13--14--15--16--17
#  |   |   |   |   |   |
#  6---7---8---9---10--11
#  |   |   |   |   |   |
#  0---1---2---3---4---5
# nodes = 6 * 6 = 36 (no change)
# edges = 2 * (5 * 6) = 60 (no change)
# faces = 5 * 5 - 1 = 24
#
# Case 4:
#  30--31--32--33--34--35
#  |   |   |   |   |   |
#  24--25--26--27--28--29
#  |   |           |   |
#  18--19          22--23
#  |   |           |   |
#  12--13          16--17
#  |   |           |   |
#  6---7---8---9---10--11
#  |   |   |   |   |   |
#  0---1---2---3---4---5
# nodes = 6 * 6 - 4 (central cell) = 32
# faces = 25 - (3 * 3) = 16
#
# Case 5:
#  30--31--32--33--34--35
#  |   |   |   |   |   |
#  24--25--26--27--28--29
#  |   |           |   |
#  18--19          22--23
#  |   |           |   |
#  12--13          16--17
#  |   |           |   |
#  6---7---8---9---10--11
#  |   |   |   |   |   |
#  0---1---2---3---4---5
# nodes = 6 * 6 - 4 (central cell) = 32
# faces = 25 - (3 * 3) = 16
#
# Case 6:
# only the 9 central cells

#  25--26--27--28
#  |   |   |   |
#  19--20--21--22
#  |   | / |   |
#  13--14--15--16
#  |   |   |   |
#  7---8---9---10
# nodes = 16
# faces = 9

cases_mesh2d_delete_small_polygon = [
    (
        True,
        DeleteMeshOption.INSIDE_NOT_INTERSECTED,
        4,
        4,
        1,
    ),
    (
        True,
        DeleteMeshOption.INSIDE_AND_INTERSECTED,
        16,
        24,
        9,
    ),
    (
        False,
        DeleteMeshOption.INSIDE_NOT_INTERSECTED,
        36,
        60,
        25,  # this is incorrect
    ),
    (
        False,
        DeleteMeshOption.INSIDE_AND_INTERSECTED,
        32,
        48,
        16,
    ),
    (
        False,
        DeleteMeshOption.FACES_WITH_INCLUDED_CIRCUMCENTERS,
        32,
        48,
        16,
    ),
    (
        True,
        DeleteMeshOption.FACES_WITH_INCLUDED_CIRCUMCENTERS,
        16,
        24,
        9,
    ),
]


@pytest.mark.parametrize(
    "invert_deletion, delete_option, exp_nodes, exp_edges, exp_faces",
    cases_mesh2d_delete_small_polygon,
)
def test_mesh2d_delete_small_polygon(
    meshkernel_with_mesh2d: MeshKernel,
    invert_deletion: bool,
    delete_option: DeleteMeshOption,
    exp_nodes: int,
    exp_edges: int,
    exp_faces: int,
):
    """Test `mesh2d_delete` by deleting a polygon from a 5x5 mesh2d.

    30--31--32--33--34--35
    |   |   |   |   |   |
    24--25--26--27--28--29
    |   | * |   | * |   |
    18--19--20--21--22--23
    |   |   |   |   |   |
    12--13--14--15--16--17
    |   | * |   | * |   |
    6---7---8---9---10--11
    |   |   |   |   |   |
    0---1---2---3---4---5

    """
    mk = meshkernel_with_mesh2d(5, 5)

    # Polygon around nodes 14, 15, 21 & 20 (through the face circum centers)
    x_coordinates = np.array([1.5, 3.5, 3.5, 1.5, 1.5], dtype=np.double)
    y_coordinates = np.array([1.5, 1.5, 3.5, 3.5, 1.5], dtype=np.double)

    geometry_list = GeometryList(x_coordinates, y_coordinates)

    mk.mesh2d_delete(geometry_list, delete_option, invert_deletion)
    mesh2d = mk.mesh2d_get()

    assert mesh2d.node_x.size == exp_nodes
    assert mesh2d.edge_x.size == exp_edges
    assert mesh2d.face_x.size == exp_faces


cases_mesh2d_delete_empty_polygon = [(False, 0, 0, 0), (True, 25, 40, 16)]


@pytest.mark.parametrize(
    "invert_deletion, exp_nodes, exp_edges, exp_faces",
    cases_mesh2d_delete_empty_polygon,
)
def test_mesh2d_delete_empty_polygon(
    meshkernel_with_mesh2d: MeshKernel,
    invert_deletion: bool,
    exp_nodes: int,
    exp_edges: int,
    exp_faces: int,
):
    """Test `mesh2d_delete` by deleting a an empty polygon from a 4x4 mesh2d.

    20--21--22--23--24
    |   |   |   |   |
    15--16--17--18--19
    |   |   |   |   |
    10--11--12--13--14
    |   |   |   |   |
    5---6---7---8---9
    |   |   |   |   |
    0---1---2---3---4

    """
    mk = meshkernel_with_mesh2d(4, 4)

    x_coordinates = np.empty(0, dtype=np.double)
    y_coordinates = np.empty(0, dtype=np.double)

    geometry_list = GeometryList(x_coordinates, y_coordinates)
    delete_option = DeleteMeshOption.INSIDE_NOT_INTERSECTED

    mk.mesh2d_delete(geometry_list, delete_option, invert_deletion)
    mesh2d = mk.mesh2d_get()

    assert mesh2d.node_x.size == exp_nodes
    assert mesh2d.edge_x.size == exp_edges
    assert mesh2d.face_x.size == exp_faces


cases_mesh2d_get_hanging_edges = [
    (
        np.array([0.0, 1.0, 1.0, 0.0], dtype=np.double),  # node_x
        np.array([0.0, 0.0, 1.0, 1.0], dtype=np.double),  # node_y
        np.array([0, 1, 1, 3, 2, 3, 2, 0], dtype=np.int32),  # edge_nodes
        np.array([], dtype=np.int32),  # expected
    ),
    (
        np.array([0.0, 1.0, 1.0, 0.0, 0.0], dtype=np.double),  # node_x
        np.array([0.0, 0.0, 1.0, 1.0, 2.0], dtype=np.double),  # node_y
        np.array([0, 1, 1, 3, 2, 3, 2, 0, 3, 4], dtype=np.int32),  # edge_nodes
        np.array([4], dtype=np.int32),  # expected
    ),
    (
        np.array([0.0, 1.0, 1.0, 0.0, 0.0, 2.0], dtype=np.double),  # node_x
        np.array([0.0, 0.0, 1.0, 1.0, 2.0, 1.0], dtype=np.double),  # node_y
        np.array([0, 1, 1, 3, 2, 3, 2, 0, 3, 4, 2, 5], dtype=np.int32),  # edge_nodes
        np.array([4, 5], dtype=np.int32),  # expected
    ),
]


@pytest.mark.parametrize(
    "node_x, node_y, edge_nodes, expected", cases_mesh2d_get_hanging_edges
)
def test_mesh2d_get_hanging_edges(
    node_x: np.ndarray, node_y: np.ndarray, edge_nodes: np.ndarray, expected: int
):
    """Tests `mesh2d_get_hanging_edges` by comparing the returned hanging edges with the expected ones
    4*
    |
    3---2---5*
    |   |
    0---1
    """

    mk = MeshKernel()

    mesh2d = Mesh2d(node_x, node_y, edge_nodes)

    mk.mesh2d_set(mesh2d)

    result = mk.mesh2d_get_hanging_edges()

    assert_array_equal(result, expected)


def test_mesh2d_delete_hanging_edges():
    """Tests `mesh2d_delete_hanging_edges` by deleting 2 hanging edges in a simple Mesh2d
    4*
    |
    3---2---5*
    |   |
    0---1
    """

    mk = MeshKernel()

    node_x = np.array([0.0, 1.0, 1.0, 0.0, 0.0, 2.0], dtype=np.double)
    node_y = np.array([0.0, 0.0, 1.0, 1.0, 2.0, 1.0], dtype=np.double)
    edge_nodes = np.array([0, 1, 1, 2, 2, 3, 3, 0, 3, 4, 2, 5], dtype=np.int32)

    mesh2d = Mesh2d(node_x, node_y, edge_nodes)

    mk.mesh2d_set(mesh2d)

    mk.mesh2d_delete_hanging_edges()

    mesh2d = mk.mesh2d_get()

    assert mesh2d.node_x.size == 4
    assert mesh2d.edge_x.size == 4
    assert mesh2d.face_x.size == 1


def test_mesh2d_make_global():
    """Tests `mesh2d_make_global`, generating a global mesh"""

    mk = MeshKernel(ProjectionType.SPHERICAL)

    num_longitude_nodes = 19
    num_latitude_nodes = 25

    mk.mesh2d_make_global(num_longitude_nodes, num_latitude_nodes)
    mesh2d = mk.mesh2d_get()

    assert mesh2d.edge_x.size == 1225
    assert mesh2d.node_x.size == 621


def test_mesh2d_make_global_with_cartesian_coordinates_should_throw():
    """Tests `mesh2d_make_global`, generating a global mesh"""

    mk = MeshKernel(ProjectionType.CARTESIAN)

    num_longitude_nodes = 19
    num_latitude_nodes = 25

    with pytest.raises(MeshKernelError):
        mk.mesh2d_make_global(num_longitude_nodes, num_latitude_nodes)


def test_mesh2d_make_triangular_mesh_from_polygon():
    """Tests `mesh2d_make_mesh_from_polygon` by creating a mesh2d from a simple hexagon."""

    mk = MeshKernel()

    #   5__4
    #  /    \
    # 0      3
    #  \1__2/
    x_coordinates = np.array([0.0, 0.5, 1.5, 2.0, 1.5, 0.5, 0.0], dtype=np.double)
    y_coordinates = np.array([1.0, 0.0, 0.0, 1.0, 2.0, 2.0, 1.0], dtype=np.double)
    polygon = GeometryList(x_coordinates, y_coordinates)

    mk.mesh2d_make_triangular_mesh_from_polygon(polygon)

    mesh2d = mk.mesh2d_get()

    assert mesh2d.node_x.size == 7
    assert mesh2d.edge_x.size == 12
    assert mesh2d.face_x.size == 6


def test_mesh2d_make_triangular_mesh_from_samples():
    """Tests `mesh2d_make_mesh_from_samples` by creating a mesh2d from six sample points."""

    mk = MeshKernel()

    #  5  4
    # 0    3
    #  1  2
    x_coordinates = np.array([0.0, 0.5, 1.5, 2.0, 1.5, 0.5, 0.0], dtype=np.double)
    y_coordinates = np.array([1.0, 0.0, 0.0, 1.0, 2.0, 2.0, 1.0], dtype=np.double)
    polygon = GeometryList(x_coordinates, y_coordinates)

    mk.mesh2d_make_triangular_mesh_from_samples(polygon)

    mesh2d = mk.mesh2d_get()

    assert mesh2d.node_x.size == 6
    assert mesh2d.edge_x.size == 9
    assert mesh2d.face_x.size == 4


def test_mesh2d_make_rectangular_mesh():
    """Tests `mesh2d_make_rectangular_mesh`."""

    make_grid_parameters = MakeGridParameters()
    make_grid_parameters.num_columns = 3
    make_grid_parameters.num_rows = 3
    make_grid_parameters.origin_x = 0.0
    make_grid_parameters.origin_y = 0.0
    make_grid_parameters.block_size_x = 10.0
    make_grid_parameters.block_size_y = 10.0

    mk_1 = MeshKernel()
    mk_1.mesh2d_make_rectangular_mesh(make_grid_parameters)
    mesh2d_1 = mk_1.mesh2d_get()

    mk_2 = MeshKernel()
    mk_2.curvilinear_compute_rectangular_grid(make_grid_parameters)
    mk_2.curvilinear_convert_to_mesh2d()
    mesh2d_2 = mk_2.mesh2d_get()

    assert mesh2d_1.node_x.size == 16
    assert mesh2d_1.edge_x.size == 24
    assert mesh2d_1.face_x.size == 9

    assert mesh2d_1 == mesh2d_2


def test_mesh2d_make_rectangular_mesh_from_polygon():
    """Tests `mesh2d_make_rectangular_mesh_from_polygon`."""

    make_grid_parameters = MakeGridParameters()
    make_grid_parameters.num_columns = 3
    make_grid_parameters.num_rows = 3
    make_grid_parameters.angle = 0.0
    make_grid_parameters.origin_x = 0.0
    make_grid_parameters.origin_y = 0.0
    make_grid_parameters.block_size_x = 1.0
    make_grid_parameters.block_size_y = 1.0

    node_x = np.array([2.5, 5.5, 3.5, 0.5, 2.5], dtype=np.double)
    node_y = np.array([0.5, 3.0, 5.0, 2.5, 0.5], dtype=np.double)
    geometry_list = GeometryList(node_x, node_y)

    mk_1 = MeshKernel()
    mk_1.mesh2d_make_rectangular_mesh_from_polygon(make_grid_parameters, geometry_list)
    mesh2d_1 = mk_1.mesh2d_get()

    mk_2 = MeshKernel()
    mk_2.curvilinear_compute_rectangular_grid_from_polygon(
        make_grid_parameters, geometry_list
    )
    mk_2.curvilinear_convert_to_mesh2d()
    mesh2d_2 = mk_2.mesh2d_get()

    assert mesh2d_1.node_x.size == 9
    assert mesh2d_1.edge_x.size == 12
    assert mesh2d_1.face_x.size == 4

    assert mesh2d_1 == mesh2d_2


def test_mesh2d_make_rectangular_mesh_on_extension():
    """Tests `mesh2d_make_rectangular_mesh_on_extension`."""

    make_grid_parameters = MakeGridParameters()
    make_grid_parameters.origin_x = -1.0
    make_grid_parameters.origin_y = 49.1
    make_grid_parameters.upper_right_x = -0.2
    make_grid_parameters.upper_right_y = 49.6
    make_grid_parameters.block_size_x = 0.01
    make_grid_parameters.block_size_y = 0.01

    mk_1 = MeshKernel(projection=ProjectionType.SPHERICAL)
    mk_1.mesh2d_make_rectangular_mesh_on_extension(make_grid_parameters)
    mesh2d_1 = mk_1.mesh2d_get()

    mk_2 = MeshKernel(projection=ProjectionType.SPHERICAL)
    mk_2.curvilinear_compute_rectangular_grid_on_extension(make_grid_parameters)
    mk_2.curvilinear_convert_to_mesh2d()
    mesh2d_2 = mk_2.mesh2d_get()

    assert mesh2d_1.node_x.size == 8343
    assert mesh2d_1.edge_x.size == 16502
    assert mesh2d_1.face_x.size == 8160

    assert mesh2d_1 == mesh2d_2


cases_polygon_refine = [
    (0, 0, 30.0, 9),
    (0, 1, 30.0, 6),
    (0, 2, 30.0, 7),
    (0, 3, 30.0, 8),
    (0, 4, 30.0, 9),
    (0, 0, 20.0, 13),
    (0, 1, 20.0, 7),
    (0, 2, 20.0, 9),
    (0, 3, 20.0, 11),
    (0, 4, 20.0, 13),
]


@pytest.mark.parametrize("start, end, length, exp_nodes", cases_polygon_refine)
def test_polygon_refine(start: int, end: int, length: float, exp_nodes: int):
    """Tests `polygon_refine` by refining a simple polygon."""

    mk = MeshKernel()

    # 3---2
    # |   |
    # 0---1
    x_coordinates = np.array([0.0, 60.0, 60.0, 0.0, 0.0], dtype=np.double)
    y_coordinates = np.array([0.0, 0.0, 60.0, 60.0, 0.0], dtype=np.double)
    polygon = GeometryList(x_coordinates, y_coordinates)

    geom = mk.polygon_refine(polygon, start, end, length)

    assert geom.x_coordinates.size == exp_nodes


cases_mesh2d_refine_based_on_samples = [(0.5, 0, 9, 12, 4), (0.5, 1, 25, 40, 16)]


@pytest.mark.parametrize(
    "min_face_size, sample_value, exp_nodes, exp_edges, exp_faces",
    cases_mesh2d_refine_based_on_samples,
)
def test_mesh2d_refine_based_on_samples(
    meshkernel_with_mesh2d: MeshKernel,
    min_face_size: float,
    sample_value: float,
    exp_nodes: int,
    exp_edges: int,
    exp_faces: int,
):
    """Tests `mesh2d_refine_based_on_samples` with a simple 2x2 mesh.

    6---7---8
    |   |   |
    3---4---5
    |   |   |
    0---1---2
    """
    mk = meshkernel_with_mesh2d(2, 2)

    x_coordinates = np.array([0.5, 0.5, 1.5, 1.5], dtype=np.double)
    y_coordinates = np.array([0.5, 1.5, 1.5, 0.5], dtype=np.double)
    values = np.array(
        [sample_value, sample_value, sample_value, sample_value], dtype=np.double
    )
    samples = GeometryList(x_coordinates, y_coordinates, values)

    refinement_params = MeshRefinementParameters(
        False, False, min_face_size, RefinementType.REFINEMENT_LEVELS, False, False, 1
    )

    mk.mesh2d_refine_based_on_samples(samples, 1.0, 1, refinement_params)

    mesdh2d = mk.mesh2d_get()

    assert mesdh2d.node_x.size == exp_nodes
    assert mesdh2d.edge_x.size == exp_edges
    assert mesdh2d.face_x.size == exp_faces


def test_refine_ridges_based_on_gridded_samples(meshkernel_with_mesh2d: MeshKernel):
    """Tests `mkernel_mesh2d_refine_ridges_based_on_gridded_samples` with a simple 5x4 mesh."""
    num_rows = 21
    num_columns = 41
    mk = meshkernel_with_mesh2d(
        rows=num_rows, columns=num_columns, spacing_x=100.0, spacing_y=100.0
    )

    refinement_params = MeshRefinementParameters(
        refine_intersected=False,
        use_mass_center_when_refining=False,
        min_edge_size=2.0,
        connect_hanging_nodes=True,
        account_for_samples_outside_face=False,
        refinement_type=RefinementType.RIDGE_DETECTION,
        max_refinement_iterations=3,
        smoothing_iterations=0,
    )

    num_sample_x_coordinates = (num_columns - 1) * 2 + 1
    num_sample_y_coordinates = (num_rows * 1) * 2 + 1

    gridded_samples = GriddedSamples(
        num_x=num_sample_x_coordinates,
        num_y=num_sample_y_coordinates,
        x_origin=0.0,
        y_origin=-0.0,
        cell_size=5.0,
        values=np.array(
            [-0.05] * num_sample_x_coordinates * num_sample_y_coordinates,
            dtype=np.float32,
        ),
    )

    relative_search_radius = 1.01
    minimum_num_samples = 1
    number_of_smoothing_iterations = 0

    mk.mesh2d_refine_ridges_based_on_gridded_samples(
        gridded_samples=gridded_samples,
        relative_search_radius=relative_search_radius,
        minimum_num_samples=minimum_num_samples,
        number_of_smoothing_iterations=number_of_smoothing_iterations,
        mesh_refinement_params=refinement_params,
    )

    mesdh2d = mk.mesh2d_get()

    assert mesdh2d.node_x.size == 924
    assert mesdh2d.edge_x.size == 1784
    assert mesdh2d.face_x.size == 861


cases_mesh2d_refine_based_on_gridded_samples = [
    (
        GriddedSamples(
            num_x=7,
            num_y=6,
            x_origin=-50.0,
            y_origin=-50.0,
            cell_size=100.0,
            values=np.array([-0.05] * 42, dtype=np.float32),
        ),
        86,
        161,
        76,
    ),
    (
        GriddedSamples(
            x_coordinates=np.array(
                [-50.0, 50.0, 150.0, 250.0, 350.0, 450.0], dtype=np.double
            ),
            y_coordinates=np.array(
                [-50.0, 50.0, 150.0, 250.0, 350.0, 450.0], dtype=np.double
            ),
            values=np.array([-0.05] * 42, dtype=np.float32),
        ),
        86,
        161,
        76,
    ),
]


@pytest.mark.parametrize(
    "gridded_samples, exp_nodes, exp_edges, exp_faces",
    cases_mesh2d_refine_based_on_gridded_samples,
)
def test_mesh2d_refine_based_on_gridded_samples(
    meshkernel_with_mesh2d: MeshKernel,
    gridded_samples: GriddedSamples,
    exp_nodes: int,
    exp_edges: int,
    exp_faces: int,
):
    """Tests `mesh2d_refine_based_on_gridded_samples` with a simple 5x4 mesh."""
    mk = meshkernel_with_mesh2d(rows=5, columns=4, spacing_x=100.0, spacing_y=100.0)

    refinement_params = MeshRefinementParameters(
        refine_intersected=False,
        use_mass_center_when_refining=False,
        min_edge_size=2.0,
        refinement_type=RefinementType.WAVE_COURANT,
        connect_hanging_nodes=True,
        account_for_samples_outside_face=False,
        max_refinement_iterations=5,
        smoothing_iterations=0,
        max_courant_time=120.0,
        directional_refinement=0,
    )

    mk.mesh2d_refine_based_on_gridded_samples(gridded_samples, refinement_params, True)

    mesdh2d = mk.mesh2d_get()

    assert mesdh2d.node_x.size == exp_nodes
    assert mesdh2d.edge_x.size == exp_edges
    assert mesdh2d.face_x.size == exp_faces


cases_mesh2d_refine_based_on_polygon = [
    (1, 25, 40, 16),
    (2, 81, 144, 64),
    (3, 289, 544, 256),
]


@pytest.mark.parametrize(
    "max_iterations, exp_nodes, exp_edges, exp_faces",
    cases_mesh2d_refine_based_on_polygon,
)
def test_mesh2d_refine_based_on_polygon(
    meshkernel_with_mesh2d: MeshKernel,
    max_iterations: int,
    exp_nodes: int,
    exp_edges: int,
    exp_faces: int,
):
    """Tests `mesh2d_refine_based_on_polygon` with a simple 2x2 mesh.

    6---7---8
    |   |   |
    3---4---5
    |   |   |
    0---1---2
    """

    mk = meshkernel_with_mesh2d(2, 2)

    x_coordinates = np.array([0.0, 0.0, 2.0, 2.0, 0.0], dtype=np.double)
    y_coordinates = np.array([0.0, 2.0, 2.0, 0.0, 0.0], dtype=np.double)
    polygon = GeometryList(x_coordinates, y_coordinates)

    refinement_params = MeshRefinementParameters(
        True, False, 0.1, 1, False, False, max_iterations
    )

    mk.mesh2d_refine_based_on_polygon(polygon, refinement_params)

    mesdh2d = mk.mesh2d_get()

    assert mesdh2d.node_x.size == exp_nodes
    assert mesdh2d.edge_x.size == exp_edges
    assert mesdh2d.face_x.size == exp_faces


def test_remove_disconnected_regions():
    """Tests `mkernel_mesh2d_remove_disconnected_regions` removes the smallest disconnected mesh"""

    mk = MeshKernel()

    # create the first mesh
    make_grid_parameters = MakeGridParameters()
    make_grid_parameters.origin_x = 0.0
    make_grid_parameters.origin_y = 0.0
    make_grid_parameters.num_rows = 10
    make_grid_parameters.num_columns = 10
    make_grid_parameters.block_size_x = 10.0
    make_grid_parameters.block_size_y = 10.0

    mk.curvilinear_compute_rectangular_grid(make_grid_parameters)
    mk.curvilinear_convert_to_mesh2d()

    # create the second mesh
    make_grid_parameters.origin_x = -100.0
    make_grid_parameters.origin_y = -100.0
    make_grid_parameters.num_rows = 3
    make_grid_parameters.num_columns = 13
    make_grid_parameters.block_size_x = 10.0
    make_grid_parameters.block_size_y = 10.0

    mk.curvilinear_compute_rectangular_grid(make_grid_parameters)
    mk.curvilinear_convert_to_mesh2d()

    mesh2d = mk.mesh2d_get()

    assert mesh2d.node_x.size == 177
    assert mesh2d.edge_x.size == 314
    assert mesh2d.face_x.size == 139

    mk.mesh2d_remove_disconnected_regions()

    mesh2d = mk.mesh2d_get()

    assert mesh2d.node_x.size == 121
    assert mesh2d.edge_x.size == 220
    assert mesh2d.face_x.size == 100


def test_mesh2d_rotate():
    """Tests `mesh2d_rotate`."""

    mk = MeshKernel()

    # set grid parameters
    make_grid_parameters = MakeGridParameters()
    make_grid_parameters.num_columns = 3
    make_grid_parameters.num_rows = 3
    make_grid_parameters.origin_x = 0.0
    make_grid_parameters.origin_y = 0.0
    make_grid_parameters.block_size_x = 1.0
    make_grid_parameters.block_size_y = 1.0

    # create cartesian grid
    mk.mesh2d_make_rectangular_mesh(make_grid_parameters)
    mesh2d = mk.mesh2d_get()

    # rotate the mesh about the origin with a small offset
    rotation_angle = -30.0  # in degrees
    rotation_origin_x = make_grid_parameters.origin_x + 1.0
    rotation_origin_y = make_grid_parameters.origin_y + 2.0
    mk.mesh2d_rotate(rotation_origin_x, rotation_origin_y, rotation_angle)
    mesh2d_rotated = mk.mesh2d_get()

    # compute expected outcome
    node_x_expected, node_y_expected = rotate(
        [mesh2d.node_x, mesh2d.node_y],
        [rotation_origin_x, rotation_origin_y],
        rotation_angle,
    )

    assert_array_almost_equal(node_x_expected, mesh2d_rotated.node_x)
    assert_array_almost_equal(node_y_expected, mesh2d_rotated.node_y)


def test_mesh2d_translate():
    """Tests `mesh2d_rotate`."""

    mk = MeshKernel()

    # set grid parameters
    make_grid_parameters = MakeGridParameters()
    make_grid_parameters.num_columns = 3
    make_grid_parameters.num_rows = 3
    make_grid_parameters.origin_x = 0.0
    make_grid_parameters.origin_y = 0.0
    make_grid_parameters.block_size_x = 1.0
    make_grid_parameters.block_size_y = 1.0

    # create cartesian grid
    mk.mesh2d_make_rectangular_mesh(make_grid_parameters)
    mesh2d = mk.mesh2d_get()

    # translate the mesh
    translation_x = 10.0
    translation_y = 15.0
    mk.mesh2d_translate(translation_x, translation_y)
    mesh2d_translated = mk.mesh2d_get()

    # compute expected outcome
    node_x_expected, node_y_expected = translate(
        [mesh2d.node_x, mesh2d.node_y], [translation_x, translation_y]
    )

    assert_array_almost_equal(node_x_expected, mesh2d_translated.node_x)
    assert_array_almost_equal(node_y_expected, mesh2d_translated.node_y)


def test_mesh2d_get_mesh_boundaries_as_polygons(meshkernel_with_mesh2d: MeshKernel):
    """Tests `mesh2d_get_mesh_boundaries_as_polygons` by checking if the resulted boundary is as expected"""

    mk = meshkernel_with_mesh2d(2, 2)

    mesh_boundary = mk.mesh2d_get_mesh_boundaries_as_polygons()
    assert_array_equal(
        mesh_boundary.x_coordinates,
        np.array([0.0, 0.0, 0.0, 1.0, 2.0, 2.0, 2.0, 1.0, 0.0], dtype=np.double),
    )
    assert_array_equal(
        mesh_boundary.y_coordinates,
        np.array([0.0, 1.0, 2.0, 2.0, 2.0, 1.0, 0.0, 0.0, 0.0], dtype=np.double),
    )


def test_mesh2d_merge_nodes():
    """Test if `mesh2d_merge_nodes` reduces the number of close nodes

    4---3
    |   |
    01--2
    """
    mk = MeshKernel()

    # Set up mesh
    edge_nodes = np.array([0, 1, 1, 2, 2, 3, 3, 4, 4, 0], dtype=np.int32)
    node_x = np.array([0.0, 1e-3, 1.0, 1.0, 0.0], dtype=np.double)
    node_y = np.array([0.0, 0.0, 0.0, 1.0, 1.0], dtype=np.double)
    input_mesh2d = Mesh2d(node_x, node_y, edge_nodes)
    mk.mesh2d_set(input_mesh2d)

    # Define polygon where we want to merge
    x_coordinates = np.array([-1.0, 2.0, 2.0, -1.0, -1.0], dtype=np.double)
    y_coordinates = np.array([-1.0, -1.0, 2.0, 2.0, -1.0], dtype=np.double)
    geometry_list = GeometryList(x_coordinates, y_coordinates)

    mk.mesh2d_merge_nodes(geometry_list)

    output_mesh2d = mk.mesh2d_get()

    assert output_mesh2d.node_x.size == 5


cases_mesh2d_merge_nodes_with_merging_distance = [(1e-2, 4), (1e-4, 5)]


@pytest.mark.parametrize(
    "merging_distance, number_of_nodes", cases_mesh2d_merge_nodes_with_merging_distance
)
def test_mesh2d_merge_nodes_with_merging_distance(
    merging_distance: float, number_of_nodes: int
):
    """Test if `mesh2d_merge_nodes` reduces the number of close nodes

    4---3
    |   |
    01--2
    """
    mk = MeshKernel()

    # Set up mesh
    edge_nodes = np.array([0, 1, 1, 2, 2, 3, 3, 4, 4, 0], dtype=np.int32)
    node_x = np.array([0.0, 1e-3, 1.0, 1.0, 0.0], dtype=np.double)
    node_y = np.array([0.0, 0.0, 0.0, 1.0, 1.0], dtype=np.double)
    input_mesh2d = Mesh2d(node_x, node_y, edge_nodes)
    mk.mesh2d_set(input_mesh2d)

    # Define polygon where we want to merge
    x_coordinates = np.array([-1.0, 2.0, 2.0, -1.0, -1.0], dtype=np.double)
    y_coordinates = np.array([-1.0, -1.0, 2.0, 2.0, -1.0], dtype=np.double)
    geometry_list = GeometryList(x_coordinates, y_coordinates)

    mk.mesh2d_merge_nodes_with_merging_distance(geometry_list, merging_distance)

    output_mesh2d = mk.mesh2d_get()

    assert output_mesh2d.node_x.size == number_of_nodes


cases_mesh2d_merge_two_nodes = [(0, 1, 4), (4, 5, 4), (0, 4, 3)]


@pytest.mark.parametrize(
    "first_node, second_node, num_faces", cases_mesh2d_merge_two_nodes
)
def test_mesh2d_merge_two_nodes(
    meshkernel_with_mesh2d: MeshKernel,
    first_node: int,
    second_node: int,
    num_faces: int,
):
    """Tests `mesh2d_merge_two_nodes` by checking if two selected nodes are properly merged

    6---7---8
    |   |   |
    3---4---5
    |   |   |
    0---1---2
    """

    mk = meshkernel_with_mesh2d(2, 2)

    mk.mesh2d_merge_two_nodes(first_node, second_node)

    output_mesh2d = mk.mesh2d_get()

    assert output_mesh2d.node_x.size == 8
    assert output_mesh2d.face_x.size == num_faces


cases_polygon_get_included_points = [
    (
        # Select all
        np.array([0.0, 3.0, 3.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 3.0, 3.0, 0.0]),
        np.array([1.0, 1.0, 1.0, 1.0, 1.0]),
    ),
    (
        # Select right half
        np.array([1.5, 3.0, 3.0, 1.5, 1.5]),
        np.array([0.0, 0.0, 3.0, 3.0, 0.0]),
        np.array([0.0, 1.0, 1.0, 0.0, 0.0]),
    ),
    (
        # Select bottom-right
        np.array([1.5, 3.0, 3.0, 1.5, 1.5]),
        np.array([0.0, 0.0, 1.5, 1.5, 0.0]),
        np.array([0.0, 1.0, 0.0, 0.0, 0.0]),
    ),
    (
        # Select top half
        np.array([0.0, 3.0, 3.0, 0.0, 0.0]),
        np.array([1.5, 1.5, 3.0, 3.0, 1.5]),
        np.array([0.0, 0.0, 1.0, 1.0, 0.0]),
    ),
    (
        # Select top-left
        np.array([0.0, 1.5, 1.5, 0.0, 0.0]),
        np.array([1.5, 1.5, 3.0, 3.0, 1.5]),
        np.array([0.0, 0.0, 0.0, 1.0, 0.0]),
    ),
]


@pytest.mark.parametrize(
    "selecting_x, selecting_y, exp_values",
    cases_polygon_get_included_points,
)
def test_polygon_get_included_points(
    selecting_x: np.array, selecting_y: np.array, exp_values: np.array
):
    """Tests `polygon_get_included_points` with a simple polygon and various selecting polygons."""

    selecting_polygon = GeometryList(selecting_x, selecting_y)

    x_coordinates = np.array([1.0, 2.0, 2.0, 1.0, 1.0], dtype=np.double)
    y_coordinates = np.array([1.0, 1.0, 2.0, 2.0, 1.0], dtype=np.double)
    selected_polygon = GeometryList(x_coordinates, y_coordinates)

    mk = MeshKernel()

    selection = mk.polygon_get_included_points(selecting_polygon, selected_polygon)

    assert_array_equal(selection.values, exp_values)


@pytest.mark.parametrize("triangulate", [True, False])
def test_mesh2d_flip_edges(triangulate: bool):
    """Tests `mesh2d_flip_edges` with a simple triangular mesh (heptagon)."""

    mk = MeshKernel()

    node_x = np.array([0, -8, -10, -4, 4, 10, 8, 0], dtype=np.double)
    node_y = np.array([10, 6, -2, -9, -9, -2, 6, -5], dtype=np.double)
    edge_nodes = np.array(
        [
            0,
            1,
            1,
            2,
            2,
            3,
            3,
            4,
            4,
            5,
            5,
            6,
            6,
            0,
            0,
            7,
            1,
            7,
            2,
            7,
            3,
            7,
            4,
            7,
            5,
            7,
            6,
            7,
        ],
        dtype=np.int32,
    )

    mk.mesh2d_set(Mesh2d(node_x, node_y, edge_nodes))

    polygon_x = np.array([-11, 11, 11, -11, -11], dtype=np.double)
    polygon_y = np.array([-11, -11, 11, 11, -11], dtype=np.double)
    polygon = GeometryList(polygon_x, polygon_y)

    land_boundaries_x = np.array([-10, -4, 4, 10], dtype=np.double)
    land_boundaries_y = np.array([-2, -9, -9, -2], dtype=np.double)
    land_boundaries = GeometryList(land_boundaries_x, land_boundaries_y)

    mk.mesh2d_flip_edges(triangulate, False, polygon, land_boundaries)

    mesh2d = mk.mesh2d_get()

    assert mesh2d.node_x.size == 8
    assert mesh2d.edge_x.size == 14
    assert mesh2d.face_x.size == 7


def test_mesh2d_flip_edges2_triangulate(meshkernel_with_mesh2d: MeshKernel):
    """Tests `mesh2d_flip_edges` with a simple 2x2 mesh.

    6---7---8       6---7---8
    |   |   |       | / | / |
    3---4---5  -->  3---4---5
    |   |   |       | / | / |
    0---1---2       0---1---2
    """

    mk = meshkernel_with_mesh2d(2, 2)

    mk.mesh2d_flip_edges(
        True,
        True,
        GeometryList(np.empty(0, dtype=np.double), np.empty(0, dtype=np.double)),
        GeometryList(np.empty(0, dtype=np.double), np.empty(0, dtype=np.double)),
    )

    mesh2d = mk.mesh2d_get()

    assert mesh2d.node_x.size == 9
    assert mesh2d.edge_x.size == 16
    assert mesh2d.face_x.size == 8

    assert np.all(mesh2d.nodes_per_face == 3)


def test_mesh2d_count_obtuse_triangles():
    r"""Tests `_mesh2d_count_obtuse_triangles` on a 3x3 mesh with two obtuse triangles.

    6---7---8
    | /   \ |
    3---4---5
    | \   / |
    0---1---2

    """
    mk = MeshKernel()

    # Mesh with obtuse triangles (4, 5, 7 and 1, 5, 4)
    node_x = np.array([0.0, 1.0, 2.0, 0.0, 1.5, 2.0, 0.0, 1.0, 2.0], dtype=np.double)
    node_y = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0], dtype=np.double)
    edge_nodes = np.array(
        [
            0,
            1,
            1,
            2,
            3,
            4,
            4,
            5,
            6,
            7,
            7,
            8,
            0,
            3,
            1,
            4,
            2,
            5,
            3,
            6,
            4,
            7,
            5,
            8,
            1,
            3,
            1,
            5,
            3,
            7,
            5,
            7,
        ],
        dtype=np.int32,
    )

    mk.mesh2d_set(Mesh2d(node_x, node_y, edge_nodes))

    n_obtuse_triangles = mk._mesh2d_count_obtuse_triangles()

    assert n_obtuse_triangles == 2


def test_mesh2d_get_obtuse_triangles_mass_centers():
    r"""Tests `mesh2d_get_obtuse_triangles_mass_centers` on a 3x3 mesh with two obtuse triangles.

    6---7---8
    | /   \ |
    3---4---5
    | \   / |
    0---1---2

    """
    mk = MeshKernel()

    # Mesh with obtuse triangles (4, 5, 7 and 1, 5, 4)
    node_x = np.array([0.0, 1.0, 2.0, 0.0, 1.5, 2.0, 0.0, 1.0, 2.0], dtype=np.double)
    node_y = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0], dtype=np.double)
    edge_nodes = np.array(
        [
            0,
            1,
            1,
            2,
            3,
            4,
            4,
            5,
            6,
            7,
            7,
            8,
            0,
            3,
            1,
            4,
            2,
            5,
            3,
            6,
            4,
            7,
            5,
            8,
            1,
            3,
            1,
            5,
            3,
            7,
            5,
            7,
        ],
        dtype=np.int32,
    )

    mk.mesh2d_set(Mesh2d(node_x, node_y, edge_nodes))

    obtuse_triangles = mk.mesh2d_get_obtuse_triangles_mass_centers()

    assert obtuse_triangles.x_coordinates.size == 2

    assert obtuse_triangles.x_coordinates[0] == 1.5
    assert obtuse_triangles.y_coordinates[0] == approx(0.666, 0.01)

    assert obtuse_triangles.x_coordinates[1] == 1.5
    assert obtuse_triangles.y_coordinates[1] == approx(1.333, 0.01)


cases_mesh2d_count_small_flow_edge_centers = [(0.9, 0), (1.0, 0), (1.1, 4)]


@pytest.mark.parametrize(
    "threshold, exp_int", cases_mesh2d_count_small_flow_edge_centers
)
def test_mesh2d_count_small_flow_edge_centers(threshold: float, exp_int: int):
    """Tests `_mesh2d_count_small_flow_edge_centers` with a simple 3x3 mesh with 4 small flow edges.

    6---7---8
    | 11|-12|
    3-|-4-|-5
    | 9-|-10|
    0---1---2
    """

    mk = MeshKernel()

    node_x = np.array(
        [0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.5, 1.5, 0.5, 1.5],
        dtype=np.double,
    )
    node_y = np.array(
        [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 0.5, 0.5, 1.5, 1.5],
        dtype=np.double,
    )
    edge_nodes = np.array(
        [
            0,
            1,
            1,
            2,
            3,
            4,
            4,
            5,
            6,
            7,
            7,
            8,
            0,
            3,
            1,
            4,
            2,
            5,
            3,
            6,
            4,
            7,
            5,
            8,
            9,
            10,
            11,
            12,
            9,
            11,
            10,
            12,
        ],
        dtype=np.int32,
    )

    mk.mesh2d_set(Mesh2d(node_x, node_y, edge_nodes))

    n_small_flow_edges = mk._mesh2d_count_small_flow_edge_centers(threshold)

    assert n_small_flow_edges == exp_int


def test_mesh2d_get_small_flow_edge_centers():
    """Tests `mesh2d_get_small_flow_edge_centers` with a simple 3x3 mesh with 4 small flow edges.

    6---7---8
    | 11|-12|
    3-|-4-|-5
    | 9-|-10|
    0---1---2
    """

    mk = MeshKernel()

    node_x = np.array(
        [0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.5, 1.5, 0.5, 1.5],
        dtype=np.double,
    )
    node_y = np.array(
        [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 0.5, 0.5, 1.5, 1.5],
        dtype=np.double,
    )
    edge_nodes = np.array(
        [
            0,
            1,
            1,
            2,
            3,
            4,
            4,
            5,
            6,
            7,
            7,
            8,
            0,
            3,
            1,
            4,
            2,
            5,
            3,
            6,
            4,
            7,
            5,
            8,
            9,
            10,
            11,
            12,
            9,
            11,
            10,
            12,
        ],
        dtype=np.int32,
    )

    mk.mesh2d_set(Mesh2d(node_x, node_y, edge_nodes))

    small_flow_edge_centers = mk.mesh2d_get_small_flow_edge_centers(1.1)

    assert small_flow_edge_centers.x_coordinates.size == 4

    assert small_flow_edge_centers.x_coordinates[0] == 0.5
    assert small_flow_edge_centers.y_coordinates[0] == 1.0
    assert small_flow_edge_centers.x_coordinates[1] == 1.5
    assert small_flow_edge_centers.y_coordinates[1] == 1.0
    assert small_flow_edge_centers.x_coordinates[2] == 1.0
    assert small_flow_edge_centers.y_coordinates[2] == 0.5
    assert small_flow_edge_centers.x_coordinates[3] == 1.0
    assert small_flow_edge_centers.y_coordinates[3] == 1.5


def test_mesh2d_delete_small_flow_edges_and_small_triangles_delete_small_flow_edges():
    r"""Tests `mesh2d_get_small_flow_edge_centers` with a simple mesh with one small flow link.

    3---4---5
    | 6-|-7 |
    0---1---2
    """

    mk = MeshKernel()

    node_x = np.array(
        [0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.5, 1.5],
        dtype=np.double,
    )
    node_y = np.array(
        [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.5, 0.5],
        dtype=np.double,
    )
    edge_nodes = np.array(
        [0, 1, 1, 2, 3, 4, 4, 5, 0, 3, 1, 4, 2, 5, 6, 7],
        dtype=np.int32,
    )

    mk.mesh2d_set(Mesh2d(node_x, node_y, edge_nodes))

    mk.mesh2d_delete_small_flow_edges_and_small_triangles(1.1, 0.01)

    mesh2d = mk.mesh2d_get()

    assert mesh2d.node_x.size == 8
    assert mesh2d.edge_x.size == 7
    assert mesh2d.face_x.size == 1


def test_mesh2d_delete_small_flow_edges_and_small_triangles_delete_small_triangles():
    r"""Tests `mesh2d_get_small_flow_edge_centers` with a simple mesh with one small triangle.

    3---4---5\
    |   |   | 6
    0---1---2/
    """

    mk = MeshKernel()

    node_x = np.array(
        [0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 2.1],
        dtype=np.double,
    )
    node_y = np.array(
        [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.5],
        dtype=np.double,
    )
    edge_nodes = np.array(
        [0, 1, 1, 2, 3, 4, 4, 5, 0, 3, 1, 4, 2, 5, 5, 6, 6, 2],
        dtype=np.int32,
    )

    mk.mesh2d_set(Mesh2d(node_x, node_y, edge_nodes))

    mk.mesh2d_delete_small_flow_edges_and_small_triangles(1.0, 0.01)

    mesh2d = mk.mesh2d_get()

    assert mesh2d.node_x.size == 7
    assert mesh2d.edge_x.size == 8
    assert mesh2d.face_x.size == 2


cases_nodes_in_polygons_mesh2d = [
    (
        np.array([1.5, 2.5, 2.5, 1.5, 1.5]),
        np.array([1.5, 1.5, 2.5, 2.5, 1.5]),
        True,
        1,
    ),
    (
        np.array([1.5, 2.5, 2.5, 1.5, 1.5]),
        np.array([1.5, 1.5, 2.5, 2.5, 1.5]),
        False,
        8,
    ),
    (
        np.array([]),
        np.array([]),
        True,
        9,
    ),
]


@pytest.mark.parametrize(
    "x_coordinates, y_coordinates, inside, exp_num_nodes",
    cases_nodes_in_polygons_mesh2d,
)
def test_nodes_in_polygons_mesh2d(
    meshkernel_with_mesh2d: MeshKernel,
    x_coordinates: ndarray,
    y_coordinates: ndarray,
    inside: bool,
    exp_num_nodes: int,
):
    """Tests `nodes_in_polygons_mesh2d` by checking if it returns the correct number of nodes

    6---7---8
    |   |   |
    3---4---5
    |   |   |
    0---1---2
    """

    mk = meshkernel_with_mesh2d(2, 2)
    geometry_list = GeometryList(x_coordinates, y_coordinates)
    selected_nodes = mk.mesh2d_get_nodes_in_polygons(geometry_list, inside)

    assert selected_nodes.size == exp_num_nodes


def test_connect_meshes():
    """Tests `mesh2d_connect_meshes`."""

    make_grid_parameters = MakeGridParameters()
    make_grid_parameters.num_columns = 3
    make_grid_parameters.num_rows = 3
    make_grid_parameters.origin_x = 0.0
    make_grid_parameters.origin_y = 0.0
    make_grid_parameters.block_size_x = 10.0
    make_grid_parameters.block_size_y = 10.0

    mk_existing = MeshKernel()
    mk_existing.mesh2d_make_rectangular_mesh(make_grid_parameters)
    mesh2d_existing = mk_existing.mesh2d_get()

    mk_to_connect = MeshKernel()
    # shift the origin by the width of the existing mesh and a small offset
    width = make_grid_parameters.num_columns * make_grid_parameters.block_size_x
    offset = 1.0
    make_grid_parameters.origin_x = width + offset

    mk_to_connect.mesh2d_make_rectangular_mesh(make_grid_parameters)
    mesh2d_to_connect = mk_to_connect.mesh2d_get()

    mk_existing.mesh2d_connect_meshes(mesh2d_to_connect, 0.4)
    mesh2d_existing = mk_existing.mesh2d_get()

    assert max(mesh2d_existing.node_x) == 2 * width + 1

    n_nodes_y = make_grid_parameters.num_rows + 1

    expected_sorted_node_x = np.array(np.repeat(0.0, n_nodes_y))
    expected_sorted_node_x = np.append(expected_sorted_node_x, np.repeat(10, n_nodes_y))
    expected_sorted_node_x = np.append(expected_sorted_node_x, np.repeat(20, n_nodes_y))
    expected_sorted_node_x = np.append(
        expected_sorted_node_x, np.repeat(30 + offset, n_nodes_y)
    )
    expected_sorted_node_x = np.append(
        expected_sorted_node_x, np.repeat(40 + offset, n_nodes_y)
    )
    expected_sorted_node_x = np.append(
        expected_sorted_node_x, np.repeat(50 + offset, n_nodes_y)
    )
    expected_sorted_node_x = np.append(
        expected_sorted_node_x, np.repeat(60 + offset, n_nodes_y)
    )

    assert_array_equal(np.sort(mesh2d_existing.node_x), expected_sorted_node_x)


def test_mesh2d_convert_projection():
    """Tests `mesh2d_convert_projection`."""

    mk = MeshKernel()

    # set grid parameters
    make_grid_parameters = MakeGridParameters()
    make_grid_parameters.num_columns = 10
    make_grid_parameters.num_rows = 15
    make_grid_parameters.block_size_x = 1.0
    make_grid_parameters.block_size_y = 1.5

    # create cartesian grid
    mk.mesh2d_make_rectangular_mesh(make_grid_parameters)
    mesh2d = mk.mesh2d_get()

    # set the zone string
    zone = "+proj=utm +lat_1=0.5 +lat_2=2 +n=0.5 +zone=31"

    # convert from Cartesian to spherical
    mk.mesh2d_convert_projection(ProjectionType.SPHERICAL, zone)
    assert mk.get_projection() == ProjectionType.SPHERICAL
    mesh2d_final = mk.mesh2d_get()
    assert not mesh2d.almost_equal(mesh2d_final, rtol=0)

    # round trip conversion to Cartesian
    mk.mesh2d_convert_projection(ProjectionType.CARTESIAN, zone)
    assert mk.get_projection() == ProjectionType.CARTESIAN
    mesh2d_final = mk.mesh2d_get()

    assert mesh2d.almost_equal(mesh2d_final, rtol=0)


cases_test_mesh2d_refine_based_on_gridded_samples_coastline = [
    InterpolationValues.SHORT,
    InterpolationValues.FLOAT,
    InterpolationValues.INT,
    InterpolationValues.DOUBLE,
    4,
]


@pytest.mark.parametrize(
    "interpolation_type", cases_test_mesh2d_refine_based_on_gridded_samples_coastline
)
def test_mesh2d_refine_based_on_gridded_samples_coastline(
    interpolation_type: InterpolationValues,
):
    """Tests `mesh2d_refine_based_on_gridded_samples` with real world data"""

    # set up
    lon_np = np.array(
        [
            -68.54791667,
            -68.46458333,
            -68.38125,
            -68.29791667,
            -68.21458333,
            -68.13125,
            -68.04791667,
            -67.96458333,
        ]
    )
    lat_np = np.array(
        [
            11.80208333,
            11.88541667,
            11.96875,
            12.05208333,
            12.13541667,
            12.21875,
            12.30208333,
            12.38541667,
            12.46875,
            12.55208333,
        ]
    )
    values_2d = np.array(
        [
            [-1700, -1769, -1688, -1641, -1526, -1291, -1121, -1537],
            [-1561, -1674, -1354, -757, -837, -838, -1080, -1466],
            [-1630, -1390, -710, -562, -479, -753, -1246, -1703],
            [-1553, -1446, -1147, -248, -175, -712, -1621, -1920],
            [-1503, -1380, -1080, -305, 18, -543, -1563, -2241],
            [-1477, -1571, -3, 100, 11, -891, -1521, -2446],
            [-1892, -1808, 16, -3102, -2015, -1302, -1484, -2581],
            [-2516, -2091, -1957, -2647, -1422, -1486, -2340, -2702],
            [-2689, -2353, -2614, -3612, -3058, -3017, -3181, -2848],
            [-3110, -3025, -3861, -3927, -3818, -4162, -4386, -4504],
        ]
    )

    if interpolation_type == InterpolationValues.SHORT:
        values_np = values_2d.flatten().astype(np.int16)
    elif interpolation_type == InterpolationValues.FLOAT:
        values_np = values_2d.flatten().astype(np.float32)
    elif interpolation_type == InterpolationValues.INT:
        values_np = values_2d.flatten().astype(np.int32)
    elif interpolation_type == InterpolationValues.DOUBLE:
        values_np = values_2d.flatten().astype(np.float64)
    else:
        values_np = values_2d.flatten().astype(np.float32)

    gridded_samples = GriddedSamples(
        x_coordinates=lon_np, y_coordinates=lat_np, values=values_np
    )

    lon_min, lon_max, lat_min, lat_max = -68.55, -67.9, 11.8, 12.6
    dx = dy = 0.05
    make_grid_parameters = MakeGridParameters(
        angle=0,
        origin_x=lon_min,
        origin_y=lat_min,
        upper_right_x=lon_max,
        upper_right_y=lat_max,
        block_size_x=dx,
        block_size_y=dy,
    )

    mk = MeshKernel(projection=ProjectionType.SPHERICAL)
    mk.curvilinear_compute_rectangular_grid_on_extension(make_grid_parameters)
    mk.curvilinear_convert_to_mesh2d()

    mesh_refinement_parameters = MeshRefinementParameters(
        min_edge_size=300,
        refinement_type=RefinementType.WAVE_COURANT,
        connect_hanging_nodes=True,
        smoothing_iterations=2,
        max_courant_time=120,
    )

    mk.mesh2d_refine_based_on_gridded_samples(
        gridded_samples=gridded_samples,
        mesh_refinement_params=mesh_refinement_parameters,
        use_nodal_refinement=True,
    )

    mesh2d = mk.mesh2d_get()

    node_x_expected = np.array(
        [
            -68.23749931,
            -68.23124931,
            -68.23749965,
            -68.23749969,
            -68.23124936,
            -68.24374969,
            -68.21874931,
            -68.21249964,
            -68.21249969,
            -68.21874936,
            -68.21249936,
            -68.20624969,
            -68.19374965,
            -68.18749931,
            -68.32499934,
            -68.31249934,
            -68.32499934,
            -68.33749934,
            -68.27499934,
            -68.26249934,
            -68.28749934,
            -68.25,
            -68.2,
            -68.35,
            -68.3,
            -68.25,
            -68.2,
            -68.3,
            -68.23125,
            -68.18125,
            -68.3625,
            -68.28125,
            -68.23125,
            -68.3625,
            -68.3125,
            -68.28125,
            -68.23125,
            -68.3125,
            -68.21874901,
            -68.22499901,
            -68.23124901,
            -68.26874901,
            -68.27499901,
            -68.28124901,
            -68.22499901,
            -68.21874901,
            -68.22499901,
            -68.23124901,
            -68.17499901,
            -68.18124901,
        ]
    )

    node_y_expected = np.array(
        [
            12.21541261,
            12.22142544,
            12.22743842,
            12.23946235,
            12.2454732,
            12.24547334,
            12.24547309,
            12.23946224,
            12.22743853,
            12.22142555,
            12.21541271,
            12.22142569,
            12.22142558,
            12.21541261,
            12.1973741,
            12.20939978,
            12.22142545,
            12.20939978,
            12.2454731,
            12.25749508,
            12.25749508,
            12.15527532,
            12.15527532,
            12.22142577,
            12.20338726,
            12.20338726,
            12.20338726,
            12.26951737,
            12.13723127,
            12.13723127,
            12.18534874,
            12.18534874,
            12.18534874,
            12.23345145,
            12.23345145,
            12.23345145,
            12.23345145,
            12.28153934,
            12.11316478,
            12.1191813,
            12.11316478,
            12.1612896,
            12.16730428,
            12.1612896,
            12.15527491,
            12.1612896,
            12.16730428,
            12.1612896,
            12.15527491,
            12.1612896,
        ]
    )

    assert mesh2d.node_x[600:650] == pytest.approx(node_x_expected, abs=1e-4)
    assert mesh2d.node_y[600:650] == pytest.approx(node_y_expected, abs=1e-4)

    assert len(mesh2d.node_x) == 1676
    assert len(mesh2d.edge_nodes) == 7176


def test_mesh2d_get_face_polygons():
    """Tests `mesh2d_get_face_polygons` with a real mesh"""

    edge_nodes = np.array(
        [
            0,
            4,
            1,
            5,
            2,
            6,
            3,
            7,
            4,
            8,
            5,
            9,
            7,
            10,
            8,
            11,
            10,
            12,
            0,
            1,
            1,
            2,
            2,
            3,
            5,
            6,
            6,
            7,
            8,
            9,
            9,
            10,
            11,
            12,
        ],
        dtype=np.int32,
    )
    node_x = np.array([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 3, 0, 1], dtype=np.double)

    node_y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 3, 3], dtype=np.double)

    input_mesh2d = Mesh2d(node_x, node_y, edge_nodes)
    mk = MeshKernel()
    mk.mesh2d_set(input_mesh2d)

    output_geometry_list = mk.mesh2d_get_face_polygons(5)

    # Test if edges are correctly calculated
    assert_array_equal(
        output_geometry_list.x_coordinates,
        np.array(
            [
                1.0000,
                2.0000,
                3.0000,
                3.0000,
                1.0000,
                1.0000,
                -999.0,
                0.0000,
                1.0000,
                3.0000,
                1.0000,
                0.0000,
                0.0000,
            ]
        ),
    )

    assert_array_equal(
        output_geometry_list.y_coordinates,
        np.array(
            [
                1.0000,
                1.0000,
                1.0000,
                2.0000,
                2.0000,
                1.0000,
                -999.0,
                2.0000,
                2.0000,
                2.0000,
                3.0000,
                3.0000,
                2.0000,
            ]
        ),
    )


def test_mesh2d_refine_based_on_gridded_samples_with_non_contiguos_arrays():
    """Tests `mesh2d_refine_based_on_gridded_samples` with non contiguos arrays"""

    # Set up
    upper_right_x = 6000
    upper_right_y = 6000

    x_coordinates = np.linspace(0, upper_right_x, num=7, dtype=np.float32)
    y_coordinates = np.linspace(0, upper_right_y, num=7, dtype=np.float32)
    values = np.full(len(x_coordinates) * len(y_coordinates), 2)

    x_coordinates = x_coordinates[::2]
    y_coordinates = y_coordinates[::2]
    values = values[::2]

    # Assert non contiguous coordinates
    assert not x_coordinates.flags.contiguous
    assert not y_coordinates.flags.contiguous
    assert not values.flags.contiguous

    gridded_samples = GriddedSamples(
        x_coordinates=x_coordinates, y_coordinates=y_coordinates, values=values
    )

    make_grid_parameters = MakeGridParameters(
        angle=0,
        origin_x=0,
        origin_y=0,
        upper_right_x=upper_right_x,
        upper_right_y=upper_right_y,
        block_size_x=2000,
        block_size_y=2000,
    )

    mk = MeshKernel(projection=ProjectionType.CARTESIAN)
    mk.curvilinear_compute_rectangular_grid_on_extension(make_grid_parameters)
    mk.curvilinear_convert_to_mesh2d()

    # original mesh dimension
    mesh2d = mk.mesh2d_get()
    assert len(mesh2d.node_x) == 16
    assert len(mesh2d.edge_nodes) == 48

    mesh_refinement_parameters = MeshRefinementParameters(
        min_edge_size=0.1,
        refinement_type=RefinementType.WAVE_COURANT,
        connect_hanging_nodes=True,
        smoothing_iterations=2,
        max_courant_time=12,
    )

    mk.mesh2d_refine_based_on_gridded_samples(
        gridded_samples=gridded_samples,
        mesh_refinement_params=mesh_refinement_parameters,
        use_nodal_refinement=True,
    )

    mesh2d = mk.mesh2d_get()

    # new mesh has more nodes and edges, it has been refined
    assert len(mesh2d.node_x) == 49
    assert len(mesh2d.edge_nodes) == 168


@pytest.mark.parametrize(
    "values_at_locations_functions",
    [
        lambda mk: mk.mesh2d_get_orthogonality(),
        lambda mk: mk.mesh2d_get_smoothness(),
    ],
)
def test_mesh2d_deletion_and_get_orthogonality(
    meshkernel_with_mesh2d: MeshKernel, values_at_locations_functions
):
    """Tests `mesh2d_delete` and `mesh2d_get_orthogonality` get the correct
    number of orthogonality values after deletion (consider only the un-gapped array)
    """

    mk = meshkernel_with_mesh2d(5, 5)

    values = values_at_locations_functions(mk).values
    mesh2d = mk.mesh2d_get()

    assert len(values) == len(mesh2d.edge_x)

    x_coordinates = np.array([-1.0, 1.5, 1.5, -1.0, -1.0])
    y_coordinates = np.array([-1.0, -1.0, 1.5, 1.5, -1.0])

    polygon = GeometryList(x_coordinates=x_coordinates, y_coordinates=y_coordinates)
    mk.mesh2d_delete(
        geometry_list=polygon,
        delete_option=DeleteMeshOption.INSIDE_NOT_INTERSECTED,
        invert_deletion=False,
    )

    values = values_at_locations_functions(mk).values
    mesh2d = mk.mesh2d_get()
    assert len(values) == len(mesh2d.edge_x)


cases_get_property = [
    (
        Mesh2d.Property.ORTHOGONALITY,
        Mesh2dLocation.EDGES,
        np.array(
            [
                -999.0,
                0.0,
                0.0,
                -999.0,
                -999.0,
                0.0,
                0.0,
                -999.0,
                -999.0,
                0.0,
                0.0,
                -999.0,
                -999.0,
                -999.0,
                -999.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                -999.0,
                -999.0,
                -999.0,
            ],
            dtype=np.double,
        ),
    ),
    (
        Mesh2d.Property.EDGE_LENGTHS,
        Mesh2dLocation.EDGES,
        np.array(
            [
                100.0,
                100.0,
                100.0,
                100.0,
                100.0,
                100.0,
                100.0,
                100.0,
                100.0,
                100.0,
                100.0,
                100.0,
                50.0,
                50.0,
                50.0,
                50.0,
                50.0,
                50.0,
                50.0,
                50.0,
                50.0,
                50.0,
                50.0,
                50.0,
            ],
            dtype=np.double,
        ),
    ),
]


@pytest.mark.parametrize(
    "property, location, expected_values",
    cases_get_property,
)
def test_mesh2d_get_property(
    meshkernel_with_mesh2d: MeshKernel,
    property: Mesh2d.Property,
    location: Mesh2dLocation,
    expected_values: ndarray,
):
    """Test mesh2d_get_property,
    getting the mesh2d property values
    """
    mk = meshkernel_with_mesh2d(rows=3, columns=3, spacing_x=50.0, spacing_y=100.0)

    property_list = mk.mesh2d_get_property(location, property)

    assert property_list.values == approx(expected_values, abs=1e-6)


def test_mesh2d_get_filtered_face_polygons():
    """Test mesh2d_get_filtered_face_polygons,
    getting the polygons of faces with all edges having bad orthogonality values
    """
    mk = MeshKernel()

    edge_nodes = np.array(
        [0, 1, 1, 2, 2, 3, 0, 3, 1, 4, 0, 4, 0, 5, 3, 5, 3, 6, 2, 6, 2, 7, 1, 7],
        dtype=np.int32,
    )

    node_x = np.array([57.0, 49.1, 58.9, 66.7, 48.8, 65.9, 67.0, 49.1], dtype=np.double)
    node_y = np.array([23.6, 14.0, 6.9, 16.2, 23.4, 24.0, 7.2, 6.7], dtype=np.double)

    input_mesh2d = Mesh2d(node_x, node_y, edge_nodes)
    mk.mesh2d_set(input_mesh2d)

    face_polygons = mk.mesh2d_get_filtered_face_polygons(
        Mesh2d.Property.ORTHOGONALITY, 0.04, 1.0
    )

    expected_coordinates_x = np.array([57.0, 49.1, 58.9, 66.7, 57.0], dtype=np.double)
    expected_coordinates_y = np.array([23.6, 14.0, 6.9, 16.2, 23.6], dtype=np.double)

    assert face_polygons.x_coordinates == approx(expected_coordinates_x, abs=1e-6)
    assert face_polygons.y_coordinates == approx(expected_coordinates_y, abs=1e-6)


def test_mesh2d__get_filtered_face_polygons_full_and_empty():

    make_grid_parameters = MakeGridParameters(
        angle=0,
        origin_x=46.2,
        origin_y=-18,
        upper_right_x=46.8,
        upper_right_y=-15.85,
        block_size_x=0.1,
        block_size_y=0.1,
    )

    mk = MeshKernel(projection=ProjectionType.SPHERICAL)
    mk.curvilinear_compute_rectangular_grid_on_extension(make_grid_parameters)
    mk.curvilinear_convert_to_mesh2d()

    orthogonality = mk.mesh2d_get_filtered_face_polygons(
        Mesh2d.Property.ORTHOGONALITY, 0.0, 1.0
    )
    assert orthogonality.x_coordinates.shape[0] == 527
    orthogonality = mk.mesh2d_get_filtered_face_polygons(
        Mesh2d.Property.ORTHOGONALITY, 0.1, 10.0
    )
    assert orthogonality.x_coordinates.shape[0] == 0
