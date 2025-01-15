import logging
import os
import platform
from ctypes import (
    CDLL,
    byref,
    c_char_p,
    c_double,
    c_int,
    c_size_t,
    create_string_buffer,
)
from enum import IntEnum
from pathlib import Path
from typing import Tuple

import numpy as np
from numpy import ndarray
from numpy.ctypeslib import as_ctypes

from meshkernel.c_structures import (
    CContacts,
    CCurvilinearGrid,
    CCurvilinearParameters,
    CGeometryList,
    CGriddedSamples,
    CMakeGridParameters,
    CMesh1d,
    CMesh2d,
    CMeshRefinementParameters,
    COrthogonalizationParameters,
    CSplinesToCurvilinearParameters,
)
from meshkernel.errors import InputError, MeshGeometryError, MeshKernelError
from meshkernel.py_structures import (
    AveragingMethod,
    Contacts,
    CurvilinearDirection,
    CurvilinearGrid,
    CurvilinearParameters,
    DeleteMeshOption,
    GeometryList,
    GriddedSamples,
    InterpolationType,
    MakeGridParameters,
    Mesh1d,
    Mesh2d,
    Mesh2dLocation,
    MeshRefinementParameters,
    OrthogonalizationParameters,
    ProjectionType,
    ProjectToLandBoundaryOption,
    SplinesToCurvilinearParameters,
)
from meshkernel.utils import (
    get_maximum_bounding_box_coordinates,
    to_contiguous_numpy_array,
)
from meshkernel.version import __version__

logger = logging.getLogger(__name__)


class MeshKernel:
    """This class is the entry point for interacting with the MeshKernel library"""

    def __init__(self, projection: ProjectionType = ProjectionType.CARTESIAN):
        """Constructor of MeshKernel

        Args:
            projection (ProjectionType, optional): The projection type. Default is `ProjectionType.CARTESIAN`.

        Raises:
            OSError: This gets raised in case MeshKernel is used within an unsupported OS.
        """

        # Determine OS
        system = platform.system()

        file_path = Path(__file__).parent
        if system == "Windows":
            lib_path = os.path.join(file_path, "MeshKernelApi.dll")
        elif system == "Linux":
            lib_path = os.path.join(file_path, "libMeshKernelApi.so")
        elif system == "Darwin":
            lib_path = os.path.join(file_path, "libMeshKernelApi.dylib")
        else:
            if not system:
                system = "Unknown OS"
            raise OSError("Unsupported operating system: {}".format(system))

        self.lib = CDLL(str(lib_path))

        self._exit_code = self.__get_exit_codes()

        self._allocate_state(projection)

        self._int_invalid_value = int(-1)

        invalid_value = self.mkernel_get_separator()

        self._float_invalid_value = float(invalid_value)

        self._set_undo_size(0)

    def __del__(self):
        self._deallocate_state()

    def __get_exit_codes(self):
        """Stores the backend exit codes
        Returns:
            An integer enumeration called exit_code holding the exit codes of the backend
        """
        success = c_int()
        self.lib.mkernel_get_exit_code_success(byref(success))

        meshkernel_error = c_int()
        self.lib.mkernel_get_exit_code_meshkernel_error(byref(meshkernel_error))

        not_implemented = c_int()
        self.lib.mkernel_get_exit_code_not_implemented_error(byref(not_implemented))

        algorithm_error = c_int()
        self.lib.mkernel_get_exit_code_algorithm_error(byref(algorithm_error))

        constraint_error = c_int()
        self.lib.mkernel_get_exit_code_constraint_error(byref(constraint_error))

        mesh_geometry_error = c_int()
        self.lib.mkernel_get_exit_code_mesh_geometry_error(byref(mesh_geometry_error))

        linear_algebra_error = c_int()
        self.lib.mkernel_get_exit_code_linear_algebra_error(byref(linear_algebra_error))

        range_error = c_int()
        self.lib.mkernel_get_exit_code_range_error(byref(range_error))

        stdlib_exception = c_int()
        self.lib.mkernel_get_exit_code_stdlib_exception(byref(stdlib_exception))

        unknown_exception = c_int()
        self.lib.mkernel_get_exit_code_unknown_exception(byref(unknown_exception))

        return IntEnum(
            "_exit_code",
            {
                "SUCCESS": success.value,
                "MESHKERNEL_ERROR": meshkernel_error.value,
                "NOT_IMPLEMENTED_ERROR": not_implemented.value,
                "ALGORITHM_ERROR": algorithm_error.value,
                "CONSTRAINT_ERROR": constraint_error.value,
                "MESH_GEOMETRY_ERROR": mesh_geometry_error.value,
                "LINEAR_ALGEBRA_ERROR": linear_algebra_error.value,
                "RANGE_ERROR": range_error.value,
                "STDLIB_EXCEPTION": stdlib_exception.value,
                "UNKNOWN_EXCEPTION": unknown_exception.value,
            },
        )

    def _allocate_state(self, projection: ProjectionType) -> None:
        """Creates a new empty mesh.

        Args:
            projection (ProjectionType): The projection type.
        """

        self._meshkernelid = c_int()
        self._execute_function(
            self.lib.mkernel_allocate_state,
            projection,
            byref(self._meshkernelid),
        )

    def _set_undo_size(self, undo_stack_size: int) -> None:
        """Sets the maximum size of the undo stack.

        Args:
            undo_stack_size (int): The maximum size of the undo stack.
        """
        self._execute_function(self.lib.mkernel_set_undo_size, undo_stack_size)

    def _deallocate_state(self) -> None:
        """
        Deallocate mesh state.

        This method is called by the destructor and
        should never be called manually
        """

        self._execute_function(
            self.lib.mkernel_deallocate_state,
            self._meshkernelid,
        )

    def mesh2d_set(self, mesh2d: Mesh2d) -> None:
        """Sets the two-dimensional mesh state of the MeshKernel.

        Please note that this involves a copy of the data.

        Args:
            mesh2d (Mesh2d): The input data used for setting the state.
        """
        c_mesh2d = CMesh2d.from_mesh2d(mesh2d)

        self._execute_function(
            self.lib.mkernel_mesh2d_set, self._meshkernelid, byref(c_mesh2d)
        )

    def mesh2d_add(self, mesh2d: Mesh2d) -> None:
        """Adds a two-dimensional mesh.

        Please note that this involves a copy of the data.

        Args:
            mesh2d (Mesh2d): The input data used for setting the state.
        """
        c_mesh2d = CMesh2d.from_mesh2d(mesh2d)

        self._execute_function(
            self.lib.mkernel_mesh2d_add, self._meshkernelid, byref(c_mesh2d)
        )

    def mesh2d_get(self) -> Mesh2d:
        """Gets the two-dimensional mesh state from the MeshKernel.

        Please note that this involves a copy of the data.

        Returns:
            Mesh2d: A copy of the two-dimensional mesh state.
        """

        c_mesh2d = self._mesh2d_get_dimensions()
        mesh2d = c_mesh2d.allocate_memory()
        self._execute_function(
            self.lib.mkernel_mesh2d_get_data, self._meshkernelid, byref(c_mesh2d)
        )

        mesh2d.remove_invalid_values(float_invalid_value=self._float_invalid_value)

        return mesh2d

    def _mesh2d_get_dimensions(self) -> CMesh2d:
        """For internal use only.

        Gets the Mesh2d dimensions.
        The integer parameters of the Mesh2D struct are set to the corresponding dimensions.
        The pointers must be set to correctly sized memory before passing the struct to `mesh2d_get`.

        Returns:
            Mesh2d: The Mesh2d dimensions.
        """
        c_mesh2d = CMesh2d()
        self._execute_function(
            self.lib.mkernel_mesh2d_get_dimensions, self._meshkernelid, byref(c_mesh2d)
        )
        return c_mesh2d

    def mesh2d_delete(
        self,
        geometry_list: GeometryList,
        delete_option: DeleteMeshOption,
        invert_deletion: bool,
    ) -> None:
        """Deletes a mesh in a polygon using several options.

        Args:
            geometry_list (GeometryList): The GeometryList describing the polygon where to perform the operation.
            delete_option (DeleteMeshOption): The option describing the strategy to delete the mesh.
            invert_deletion (bool): Whether or not to invert the deletion.
        """

        c_geometry_list = CGeometryList.from_geometrylist(geometry_list)

        self._execute_function(
            self.lib.mkernel_mesh2d_delete,
            self._meshkernelid,
            byref(c_geometry_list),
            c_int(delete_option),
            c_int(invert_deletion),
        )

    def mesh2d_insert_edge(self, start_node: int, end_node: int) -> int:
        """Insert a new mesh2d edge connecting two given nodes.

        Args:
            start_node (int): The index of the first node.
            end_node (int): The index of the second node.

        Returns:
            int: The index of the new edge.
        """

        edge_index = c_int()
        self._execute_function(
            self.lib.mkernel_mesh2d_insert_edge,
            self._meshkernelid,
            c_int(start_node),
            c_int(end_node),
            byref(edge_index),
        )

        return edge_index.value

    def mesh2d_insert_node(self, x: float, y: float) -> int:
        """Insert a new node at the specified coordinates

        Args:
            x (float): The x-coordinate of the new node.
            y (float): The y-coordinate of the new node.

        Returns:
            int: The index of the new node.
        """

        index = c_int()

        self._execute_function(
            self.lib.mkernel_mesh2d_insert_node,
            self._meshkernelid,
            c_double(x),
            c_double(y),
            byref(index),
        )
        return index.value

    def mesh2d_delete_node(self, node_index: int) -> None:
        """Deletes a Mesh2d node with the given `index`.

        Args:
            node_index (int): The index of the node to be deleted.

        Raises:
            InputError: Raised when `node_index` is smaller than 0.
        """

        if node_index < 0:
            raise InputError("node_index needs to be a positive integer")

        self._execute_function(
            self.lib.mkernel_mesh2d_delete_node, self._meshkernelid, c_int(node_index)
        )

    def mesh2d_move_node(self, x: float, y: float, node_index: int) -> None:
        """Moves a Mesh2d node with the given `index` to the point position.

        Args:
            x (float): The x-coordinate of the new position of the node.
            y (float): The y-coordinate of the new position of the node.
            node_index (int): The index of the node to be moved.

        Raises:
            InputError: Raised when `node_index` is smaller than 0.
        """

        if node_index < 0:
            raise InputError("node_index needs to be a positive integer")

        self._execute_function(
            self.lib.mkernel_mesh2d_move_node,
            self._meshkernelid,
            c_double(x),
            c_double(y),
            c_int(node_index),
        )

    def mesh2d_delete_edge(self, x_coordinate: float, y_coordinate: float) -> None:
        """Deletes the closest mesh2d edge to a point.
        The coordinates of the edge middle points are used for calculating the distances to the point.

        Args:
            x (float): The x-coordinate of the point.
            y (float): The y-coordinate of the point.
        """

        (
            x_lower_left,
            y_lower_left,
            x_upper_right,
            y_upper_right,
        ) = get_maximum_bounding_box_coordinates()

        self._execute_function(
            self.lib.mkernel_mesh2d_delete_edge,
            self._meshkernelid,
            c_double(x_coordinate),
            c_double(y_coordinate),
            c_double(x_lower_left),
            c_double(y_lower_left),
            c_double(x_upper_right),
            c_double(y_upper_right),
        )

    def mesh2d_get_edge(self, x: float, y: float) -> int:
        """Gets the closest mesh2d edge to a point.

        Args:
            x (float): The x-coordinate of the point.
            y (float): The y-coordinate of the point.

        Returns:
            int: The index of the edge (gap-free array index)
        """

        index = c_int()

        (
            x_lower_left,
            y_lower_left,
            x_upper_right,
            y_upper_right,
        ) = get_maximum_bounding_box_coordinates()

        self._execute_function(
            self.lib.mkernel_mesh2d_get_edge,
            self._meshkernelid,
            c_double(x),
            c_double(y),
            c_double(x_lower_left),
            c_double(y_lower_left),
            c_double(x_upper_right),
            c_double(y_upper_right),
            byref(index),
        )

        return index.value

    def mesh2d_get_face_polygons(self, num_edges: int) -> GeometryList:
        """Gets the faces polygons with a number of edges equal to num_edges.

        Args:
            num_edges (int): The number of edges

        Returns:
            GeometryList: The resulting face polygons
        """
        c_geometry_list_dimension = c_int()

        self._execute_function(
            self.lib.mkernel_mesh2d_get_face_polygons_dimension,
            self._meshkernelid,
            c_int(num_edges),
            byref(c_geometry_list_dimension),
        )

        n_coordinates = c_geometry_list_dimension.value
        x_coordinates = np.empty(n_coordinates, dtype=np.double)
        y_coordinates = np.empty(n_coordinates, dtype=np.double)

        face_polygons = GeometryList(
            x_coordinates=x_coordinates, y_coordinates=y_coordinates
        )
        c_face_polygons = CGeometryList.from_geometrylist(face_polygons)

        self._execute_function(
            self.lib.mkernel_mesh2d_get_face_polygons,
            self._meshkernelid,
            c_int(num_edges),
            byref(c_face_polygons),
        )

        return face_polygons

    def mesh2d_get_filtered_face_polygons(
        self, property: Mesh2d.Property, min_value: float, max_value: float
    ) -> GeometryList:
        """Gets the polygons matching the metric value within the minimum and maximum value.

        Args:

            property (Mesh2d.Property): The property used to filter the locations
            min_value(float): The minimum value of the metric.
            max_value(float): The maximum value of the metric.

        Returns:
            GeometryList: The resulting face polygons
        """
        c_geometry_list_dimension = c_int()

        self._execute_function(
            self.lib.mkernel_mesh2d_get_filtered_face_polygons_dimension,
            self._meshkernelid,
            c_int(property),
            c_double(min_value),
            c_double(max_value),
            byref(c_geometry_list_dimension),
        )

        n_coordinates = c_geometry_list_dimension.value
        x_coordinates = np.empty(n_coordinates, dtype=np.double)
        y_coordinates = np.empty(n_coordinates, dtype=np.double)

        face_polygons = GeometryList(
            x_coordinates=x_coordinates, y_coordinates=y_coordinates
        )
        c_face_polygons = CGeometryList.from_geometrylist(face_polygons)

        self._execute_function(
            self.lib.mkernel_mesh2d_get_filtered_face_polygons,
            self._meshkernelid,
            c_int(property),
            c_double(min_value),
            c_double(max_value),
            byref(c_face_polygons),
        )

        return face_polygons

    def mesh2d_get_node_index(self, x: float, y: float, search_radius: float) -> int:
        """Finds the node closest to a point within a given search radius.

        Args:
            x (float): The x-coordinate of the point.
            y (float): The y-coordinate of the point.
            search_radius (float): The search radius.

        Returns:
            int: The index of node (gap-free array index)
        """

        index = c_int()

        (
            x_lower_left,
            y_lower_left,
            x_upper_right,
            y_upper_right,
        ) = get_maximum_bounding_box_coordinates()

        self._execute_function(
            self.lib.mkernel_mesh2d_get_node_index,
            self._meshkernelid,
            c_double(x),
            c_double(y),
            c_double(search_radius),
            c_double(x_lower_left),
            c_double(y_lower_left),
            c_double(x_upper_right),
            c_double(y_upper_right),
            byref(index),
        )

        return index.value

    def mesh2d_get_hanging_edges(self) -> ndarray:
        """Gets the indices of hanging edges. A hanging edge is an edge where one of the two nodes is not connected.

        Returns:
            ndarray:  The integer array describing the indices of the hanging edges.
        """

        # Get number of hanging edges
        number_hanging_edges = self._mesh2d_count_hanging_edges()

        # Get hanging edges
        hanging_edges = np.empty(number_hanging_edges, dtype=np.int32)
        c_hanging_edges = np.ctypeslib.as_ctypes(hanging_edges)
        self._execute_function(
            self.lib.mkernel_mesh2d_get_hanging_edges,
            self._meshkernelid,
            byref(c_hanging_edges),
        )

        return hanging_edges

    def _mesh2d_count_hanging_edges(self) -> int:
        """For internal use only.

        Count the number of hanging edges in a mesh2d.
        An hanging edge is an edge where one of the two nodes is not connected.

        Returns:
            int: The number of hanging edges.
        """
        c_number_hanging_edges = c_int()
        self._execute_function(
            self.lib.mkernel_mesh2d_count_hanging_edges,
            self._meshkernelid,
            byref(c_number_hanging_edges),
        )
        return c_number_hanging_edges.value

    def mesh2d_delete_hanging_edges(self) -> None:
        """Delete the hanging edges in the Mesh2d.
        A hanging edge is an edge where one of the two nodes is not connected.
        """

        self._execute_function(
            self.lib.mkernel_mesh2d_delete_hanging_edges, self._meshkernelid
        )

    def mesh2d_make_global(
        self, num_longitude_nodes: int, num_latitude_nodes: int
    ) -> None:
        """Compute the global mesh with a given number of points along the longitude and latitude directions.

        Args:
        num_longitude_nodes (int): The number of points along the longitude.
        num_latitude_nodes (int):  The number of points along the latitude (half hemisphere)
        """

        self._execute_function(
            self.lib.mkernel_mesh2d_make_global,
            self._meshkernelid,
            c_int(num_longitude_nodes),
            c_int(num_latitude_nodes),
        )

    def mesh2d_make_triangular_mesh_from_polygon(self, polygon: GeometryList) -> None:
        """Generates a triangular mesh2d within a polygon. The size of the triangles is determined from the length of
        the polygon edges.

        Args:
            polygon (GeometryList): The polygon.
        """

        c_geometry_list = CGeometryList.from_geometrylist(polygon)

        self._execute_function(
            self.lib.mkernel_mesh2d_make_triangular_mesh_from_polygon,
            self._meshkernelid,
            byref(c_geometry_list),
        )

    def mesh2d_make_triangular_mesh_from_samples(
        self, sample_points: GeometryList
    ) -> None:
        """Makes a triangular mesh from a set of samples, triangulating the sample points.

        Args:
            sample_points (GeometryList): The sample points.
        """

        c_geometry_list = CGeometryList.from_geometrylist(sample_points)

        self._execute_function(
            self.lib.mkernel_mesh2d_make_triangular_mesh_from_samples,
            self._meshkernelid,
            byref(c_geometry_list),
        )

    def mesh2d_make_rectangular_mesh(
        self, make_grid_parameters: MakeGridParameters
    ) -> None:
        """Generates a rectangular mesh2d.

        Args:
            make_grid_parameters (MakeGridParameters): The parameters used for making the uniform grid
        """

        c_make_grid_parameters = CMakeGridParameters.from_makegridparameters(
            make_grid_parameters
        )

        self._execute_function(
            self.lib.mkernel_mesh2d_make_rectangular_mesh,
            self._meshkernelid,
            byref(c_make_grid_parameters),
        )

    def mesh2d_make_rectangular_mesh_from_polygon(
        self, make_grid_parameters: MakeGridParameters, polygon: GeometryList
    ) -> None:
        """Generates a rectangular mesh2d within a polygon.

        Args:
            make_grid_parameters (MakeGridParameters): The parameters used for making the uniform grid
            geometry_list (GeometryList): The polygon within which the grid will be generated
        """

        c_make_grid_parameters = CMakeGridParameters.from_makegridparameters(
            make_grid_parameters
        )

        c_geometry_list = CGeometryList.from_geometrylist(polygon)

        self._execute_function(
            self.lib.mkernel_mesh2d_make_rectangular_mesh_from_polygon,
            self._meshkernelid,
            byref(c_make_grid_parameters),
            byref(c_geometry_list),
        )

    def mesh2d_make_rectangular_mesh_on_extension(
        self, make_grid_parameters: MakeGridParameters
    ) -> None:
        """Generates a rectangular mesh2d.

        Args:
            make_grid_parameters (MakeGridParameters): The parameters used for making the uniform grid
        """

        c_make_grid_parameters = CMakeGridParameters.from_makegridparameters(
            make_grid_parameters
        )

        self._execute_function(
            self.lib.mkernel_mesh2d_make_rectangular_mesh_on_extension,
            self._meshkernelid,
            byref(c_make_grid_parameters),
        )

    def polygon_refine(
        self,
        polygon: GeometryList,
        first_node: int,
        second_node: int,
        target_edge_length: float,
    ) -> GeometryList:
        """Refines the polygon perimeter between two nodes. This interval is refined to achieve a target edge length.

        Args:
            polygon (GeometryList): The input polygon to refine.
            first_node (int): The first index of the refinement interval.
            second_node (int): The second index of the refinement interval.
            target_edge_length (float): The target interval edge length.

        Returns:
            GeometryList: The refined polygon.
        """
        c_polygon = CGeometryList.from_geometrylist(polygon)
        c_n_polygon_nodes = c_int()

        self._execute_function(
            self.lib.mkernel_polygon_count_refine,
            self._meshkernelid,
            byref(c_polygon),
            c_int(first_node),
            c_int(second_node),
            c_double(target_edge_length),
            byref(c_n_polygon_nodes),
        )

        n_coordinates = c_n_polygon_nodes.value

        x_coordinates = np.empty(n_coordinates, dtype=np.double)
        y_coordinates = np.empty(n_coordinates, dtype=np.double)
        refined_polygon = GeometryList(x_coordinates, y_coordinates)

        c_refined_polygon = CGeometryList.from_geometrylist(refined_polygon)

        self._execute_function(
            self.lib.mkernel_polygon_refine,
            self._meshkernelid,
            byref(c_polygon),
            c_int(first_node),
            c_int(second_node),
            c_double(target_edge_length),
            byref(c_refined_polygon),
        )

        return refined_polygon

    def mesh2d_refine_based_on_samples(
        self,
        samples: GeometryList,
        relative_search_radius: float,
        minimum_num_samples: int,
        mesh_refinement_params: MeshRefinementParameters,
    ) -> None:
        """Refines a mesh2d based on samples. Refinement is achieved by successive splits of the face edges.
        The number of successive splits is indicated by the sample value.
        For example:
        - a value of 0 means no split and hence no refinement;
        - a value of 1 means a single split (a quadrilateral face generates 4 faces);
        - a value of 2 two splits (a quadrilateral face generates 16 faces).

        Args:
            samples (GeometryList): The samples.
            relative_search_radius (float): The relative search radius relative to the face size,
                                            used for some interpolation algorithms.
            minimum_num_samples (int): The minimum number of samples used for some averaging algorithms.
            mesh_refinement_params (MeshRefinementParameters): The mesh refinement parameters.
        """

        c_samples = CGeometryList.from_geometrylist(samples)
        c_refinement_params = CMeshRefinementParameters.from_meshrefinementparameters(
            mesh_refinement_params
        )

        self._execute_function(
            self.lib.mkernel_mesh2d_refine_based_on_samples,
            self._meshkernelid,
            byref(c_samples),
            c_double(relative_search_radius),
            c_int(minimum_num_samples),
            byref(c_refinement_params),
        )

    def mesh2d_refine_ridges_based_on_gridded_samples(
        self,
        gridded_samples: GriddedSamples,
        relative_search_radius: float,
        minimum_num_samples: int,
        number_of_smoothing_iterations: int,
        mesh_refinement_params: MeshRefinementParameters,
    ) -> None:
        """Refines a mesh2d based on samples with ridge refinement. This method automatically detects the ridges in a sample set.

        Args:
            gridded_samples (GriddedSamples): The gridded samples.
            relative_search_radius (float): The relative search radius relative to the face size,
                                            used for some interpolation algorithms.
            minimum_num_samples (int): The minimum number of samples used for some averaging algorithms.
            number_of_smoothing_iterations (int): The number of smoothing iterations to apply to the input sample set.
            mesh_refinement_params (MeshRefinementParameters): The mesh refinement parameters.
        """

        c_gridded_samples = CGriddedSamples.from_griddedSamples(gridded_samples)
        c_refinement_params = CMeshRefinementParameters.from_meshrefinementparameters(
            mesh_refinement_params
        )

        self._execute_function(
            self.lib.mkernel_mesh2d_refine_ridges_based_on_gridded_samples,
            self._meshkernelid,
            byref(c_gridded_samples),
            c_double(relative_search_radius),
            c_int(minimum_num_samples),
            c_int(number_of_smoothing_iterations),
            byref(c_refinement_params),
        )

    def mesh2d_refine_based_on_gridded_samples(
        self,
        gridded_samples: GriddedSamples,
        mesh_refinement_params: MeshRefinementParameters,
        use_nodal_refinement: bool = True,
    ) -> None:
        """Computes mesh refinement based of gridded samples and bilinear interpolation

        Args:
            gridded_samples (GriddedSamples): The gridded samples.
            mesh_refinement_params (MeshRefinementParameters): The mesh refinement parameters.
            use_nodal_refinement (bool): If the depth value at nodes is used for refinement. Default True.
        """

        c_gridded_samples = CGriddedSamples.from_griddedSamples(gridded_samples)
        c_refinement_params = CMeshRefinementParameters.from_meshrefinementparameters(
            mesh_refinement_params
        )
        use_nodal_refinement_int = 1 if use_nodal_refinement else 0

        self._execute_function(
            self.lib.mkernel_mesh2d_refine_based_on_gridded_samples,
            self._meshkernelid,
            byref(c_gridded_samples),
            byref(c_refinement_params),
            c_int(use_nodal_refinement_int),
        )

    def mesh2d_refine_based_on_polygon(
        self,
        polygon: GeometryList,
        mesh_refinement_params: MeshRefinementParameters,
    ) -> None:
        """Refines a mesh2d within a polygon. Refinement is achieved by splitting the edges contained
        in the polygon in two.

        Args:
            samples (GeometryList): The closed polygon.
            mesh_refinement_params (MeshRefinementParameters): The mesh refinement parameters.
        """

        c_polygon = CGeometryList.from_geometrylist(polygon)
        c_refinement_params = CMeshRefinementParameters.from_meshrefinementparameters(
            mesh_refinement_params
        )

        self._execute_function(
            self.lib.mkernel_mesh2d_refine_based_on_polygon,
            self._meshkernelid,
            byref(c_polygon),
            byref(c_refinement_params),
        )

    def mesh2d_remove_disconnected_regions(
        self,
    ) -> None:
        """Remove any disconnected regions from a mesh2d. Only the most frequent remains"""

        self._execute_function(
            self.lib.mkernel_mesh2d_remove_disconnected_regions,
            self._meshkernelid,
        )

    def mesh2d_rotate(self, centre_x: float, centre_y: float, angle: float) -> None:
        """Rotates a mesh2d by about a centre of rotation.

        Args:
        centre_x (float): X-coordinate of the centre of rotation.
        centre_y (float): Y-coordinate of the centre of rotation.
        angle (float): Angle of rotation in degrees.

        """

        self._execute_function(
            self.lib.mkernel_mesh2d_rotate,
            self._meshkernelid,
            c_double(centre_x),
            c_double(centre_y),
            c_double(angle),
        )

    def mesh2d_translate(self, translation_x: float, translation_y: float) -> None:
        """Translates a mesh2d.

        Args:
        translation_x (float): X-component of the translation vector.
        translation_y (float): Y-component of the translation vector

        """

        self._execute_function(
            self.lib.mkernel_mesh2d_translate,
            self._meshkernelid,
            c_double(translation_x),
            c_double(translation_y),
        )

    def polygon_get_included_points(
        self, selecting_polygon: GeometryList, selected_polygon: GeometryList
    ) -> GeometryList:
        """Selects the polygon points within another polygon.

        Args:
            selecting_polygon (GeometryList): The selection polygon.
            selected_polygon (GeometryList): The polygon of which to get the selected points.

        Returns:
            GeometryList: The selection result. The selected points are contained in the values array of the returned
                          GeometryList (0.0 not selected, 1.0 selected).
        """

        c_selecting_polygon = CGeometryList.from_geometrylist(selecting_polygon)
        c_selected_polygon = CGeometryList.from_geometrylist(selected_polygon)

        n_coordinates = selected_polygon.x_coordinates.size

        x_coordinates = np.empty(n_coordinates, dtype=np.double)
        y_coordinates = np.empty(n_coordinates, dtype=np.double)
        values = np.empty(n_coordinates, dtype=np.double)
        selection = GeometryList(x_coordinates, y_coordinates, values)

        c_selection = CGeometryList.from_geometrylist(selection)

        self._execute_function(
            self.lib.mkernel_polygon_get_included_points,
            self._meshkernelid,
            byref(c_selecting_polygon),
            byref(c_selected_polygon),
            byref(c_selection),
        )

        return selection

    def mesh2d_flip_edges(
        self,
        triangulation_required: bool,
        project_to_land_boundary_required: bool,
        selecting_polygon: GeometryList,
        land_boundaries: GeometryList,
    ):
        """Flips mesh2d edges to optimize the mesh smoothness.
        Nodes that are connected to more than six other nodes are typically enclosed by faces of highly non-uniform
        shape and wildly varying areas.

        Args:
            triangulation_required (bool): Whether to triangulate non-triangular cells.
            project_to_land_boundary_required (bool): Whether projection to land boundaries is required.
            selecting_polygon (GeometryList): The polygon where to perform the edge flipping.
            land_boundaries (GeometryList): The land boundaries to account for when flipping the edges.

        """

        c_selecting_polygon = CGeometryList.from_geometrylist(selecting_polygon)
        c_land_boundaries = CGeometryList.from_geometrylist(land_boundaries)

        self._execute_function(
            self.lib.mkernel_mesh2d_flip_edges,
            self._meshkernelid,
            c_int(triangulation_required),
            c_int(project_to_land_boundary_required),
            byref(c_selecting_polygon),
            byref(c_land_boundaries),
        )

    def _mesh2d_count_obtuse_triangles(self) -> int:
        """For internal use only.

        Gets the number of obtuse mesh2d triangles.
        Obtuse triangles are those having one angle larger than 90°.

        Returns:
            int: The number of obtuse triangles.
        """

        n_obtuse_triangles = c_int(0)

        self._execute_function(
            self.lib.mkernel_mesh2d_count_obtuse_triangles,
            self._meshkernelid,
            byref(n_obtuse_triangles),
        )

        return n_obtuse_triangles.value

    def mesh2d_get_obtuse_triangles_mass_centers(self) -> GeometryList:
        """Gets the mass centers of obtuse mesh2d triangles.
        Obtuse triangles are those having one angle larger than 90°.

        Returns:
            GeometryList: The geometry list with the mass center coordinates.
        """
        n_obtuse_triangles = self._mesh2d_count_obtuse_triangles()

        x_coordinates = np.empty(n_obtuse_triangles, dtype=np.double)
        y_coordinates = np.empty(n_obtuse_triangles, dtype=np.double)
        geometry_list = GeometryList(x_coordinates, y_coordinates)

        c_geometry_list = CGeometryList.from_geometrylist(geometry_list)

        self._execute_function(
            self.lib.mkernel_mesh2d_get_obtuse_triangles_mass_centers,
            self._meshkernelid,
            byref(c_geometry_list),
        )

        return geometry_list

    def _mesh2d_count_small_flow_edge_centers(
        self, small_flow_edges_length_threshold: float
    ) -> int:
        """For internal use only.

        Counts the number of small mesh2d flow edges.
        The flow edges are the edges connecting face circumcenters.

        Args:
            small_flow_edges_length_threshold (float): The configurable length for detecting a small flow edge.

        Returns:
            int: The number of the small flow edges.
        """

        n_small_flow_edge_centers = c_int()
        self._execute_function(
            self.lib.mkernel_mesh2d_count_small_flow_edge_centers,
            self._meshkernelid,
            c_double(small_flow_edges_length_threshold),
            byref(n_small_flow_edge_centers),
        )

        return n_small_flow_edge_centers.value

    def mesh2d_get_small_flow_edge_centers(
        self, small_flow_edges_length_threshold: float
    ) -> GeometryList:
        """Gets the small mesh2d flow edges centers.
        The flow edges are the edges connecting face circumcenters.

        Args:
            small_flow_edges_length_threshold (float): The configurable length for detecting a small flow edge.

        Returns:
            int: The geometry list with the small flow edge center coordinates.
        """

        n_small_flow_edge_centers = self._mesh2d_count_small_flow_edge_centers(
            small_flow_edges_length_threshold
        )

        x_coordinates = np.empty(n_small_flow_edge_centers, dtype=np.double)
        y_coordinates = np.empty(n_small_flow_edge_centers, dtype=np.double)
        geometry_list = GeometryList(x_coordinates, y_coordinates)

        c_geometry_list = CGeometryList.from_geometrylist(geometry_list)

        self._execute_function(
            self.lib.mkernel_mesh2d_get_small_flow_edge_centers,
            self._meshkernelid,
            c_double(small_flow_edges_length_threshold),
            byref(c_geometry_list),
        )

        return geometry_list

    def mesh2d_delete_small_flow_edges_and_small_triangles(
        self,
        small_flow_edges_length_threshold: float,
        min_fractional_area_triangles: float,
    ) -> None:
        """Deletes all small mesh2d flow edges and small triangles.
        The flow edges are the edges connecting faces circumcenters.

        Args:
            small_flow_edges_length_threshold (float): The configurable length for detecting a small flow edge.
            min_fractional_area_triangles (float): The ratio of the face area to the average area of neighboring
                                                   non-triangular faces. This parameter is used for determining whether
                                                   a triangular face is small.
        """

        self._execute_function(
            self.lib.mkernel_mesh2d_delete_small_flow_edges_and_small_triangles,
            self._meshkernelid,
            c_double(small_flow_edges_length_threshold),
            c_double(min_fractional_area_triangles),
        )

    def get_splines(
        self, geometry_list: GeometryList, number_of_points_between_nodes: int
    ) -> GeometryList:
        """Get the computed spline points between two corner nodes.

        Args:
            geometry_list (GeometryList): The input corner nodes of the splines.
            number_of_points_between_nodes (int): The number of spline points to generate between two corner nodes.

        Returns:
            GeometryList: The output spline.
        """

        # Allocate space for output
        original_number_of_coordinates = geometry_list.x_coordinates.size
        number_of_coordinates = (
            original_number_of_coordinates * number_of_points_between_nodes
            - number_of_points_between_nodes
            + original_number_of_coordinates
            + 1
        )
        x_coordinates = np.empty(number_of_coordinates, dtype=np.double)
        y_coordinates = np.empty(number_of_coordinates, dtype=np.double)
        values = np.empty(number_of_coordinates, dtype=np.double)
        geometry_list_out = GeometryList(x_coordinates, y_coordinates, values)

        # Convert to CGeometryList
        c_geometry_list_in = CGeometryList.from_geometrylist(geometry_list)
        c_geometry_list_out = CGeometryList.from_geometrylist(geometry_list_out)

        self._execute_function(
            self.lib.mkernel_get_splines,
            byref(c_geometry_list_in),
            byref(c_geometry_list_out),
            c_int(number_of_points_between_nodes),
        )

        return geometry_list_out

    def mesh2d_get_mesh_boundaries_as_polygons(self) -> GeometryList:
        """Retrieves the boundaries of a mesh as a series of separated polygons.

        For example, if a mesh has an single inner hole, two polygons will be generated,
        one for the inner boundary and one for the outer boundary.

        Returns:
            GeometryList: The output network boundary polygon.
        """

        # Get number of polygon nodes
        number_of_polygon_nodes = self._mesh2d_count_mesh_boundaries_as_polygons()

        # Create GeometryList instance
        x_coordinates = np.empty(number_of_polygon_nodes, dtype=np.double)
        y_coordinates = np.empty(number_of_polygon_nodes, dtype=np.double)
        geometry_list_out = GeometryList(x_coordinates, y_coordinates)

        # Get mesh boundary
        c_geometry_list_out = CGeometryList.from_geometrylist(geometry_list_out)
        self._execute_function(
            self.lib.mkernel_mesh2d_get_mesh_boundaries_as_polygons,
            self._meshkernelid,
            byref(c_geometry_list_out),
        )

        return geometry_list_out

    def _mesh2d_count_mesh_boundaries_as_polygons(self) -> int:
        """For internal use only.

        Counts the number of polygon nodes contained in the mesh boundary polygons
        computed in function mesh2d_get_mesh_boundaries_as_polygons.

        Returns:
            int: The number of polygon nodes.
        """
        number_of_polygon_nodes = c_int()
        self._execute_function(
            self.lib.mkernel_mesh2d_count_mesh_boundaries_as_polygons,
            self._meshkernelid,
            byref(number_of_polygon_nodes),
        )
        return number_of_polygon_nodes.value

    def mesh2d_merge_nodes(self, geometry_list: GeometryList) -> None:
        """Merges the mesh2d nodes, effectively removing all small edges.

        Args:
            geometry_list (GeometryList): The polygon defining the area where the operation will be performed.
        """
        c_geometry_list = CGeometryList.from_geometrylist(geometry_list)
        self._execute_function(
            self.lib.mkernel_mesh2d_merge_nodes,
            self._meshkernelid,
            byref(c_geometry_list),
        )

    def mesh2d_merge_nodes_with_merging_distance(
        self, geometry_list: GeometryList, merging_distance: float
    ) -> None:
        """Merges the mesh2d nodes, effectively removing all small edges.

        Args:
            geometry_list (GeometryList): The polygon defining the area where the operation will be performed.
            merging_distance (float): The distance below which two nodes will be merged.
        """
        c_geometry_list = CGeometryList.from_geometrylist(geometry_list)
        self._execute_function(
            self.lib.mkernel_mesh2d_merge_nodes_with_merging_distance,
            self._meshkernelid,
            byref(c_geometry_list),
            c_double(merging_distance),
        )

    def mesh2d_merge_two_nodes(self, first_node: int, second_node: int) -> None:
        """Merges `first_node` into `second_node`.

        Args:
            first_node (int): The index of the first node to merge.
            second_node (int): The index of the second node to merge.
        """

        self._execute_function(
            self.lib.mkernel_mesh2d_merge_two_nodes,
            self._meshkernelid,
            c_int(first_node),
            c_int(second_node),
        )

    def mesh2d_get_nodes_in_polygons(
        self, geometry_list: GeometryList, inside: bool
    ) -> ndarray:
        """Gets the indices of the mesh2d nodes selected with a polygon.

        Args:
            geometry_list (GeometryList): The input polygon.
            inside (bool): Selection of the nodes inside the polygon (True) or outside (False).

        Returns:
            ndarray: The integer array describing the selected nodes indices.
        """

        # Get number of mesh nodes
        number_of_mesh_nodes = self._mesh2d_count_nodes_in_polygons(
            geometry_list, inside
        )

        selected_nodes = np.empty(number_of_mesh_nodes, dtype=np.int32)
        c_selected_nodes = np.ctypeslib.as_ctypes(selected_nodes)
        c_geometry_list = CGeometryList.from_geometrylist(geometry_list)

        # Get selected nodes
        self._execute_function(
            self.lib.mkernel_mesh2d_get_nodes_in_polygons,
            self._meshkernelid,
            byref(c_geometry_list),
            c_int(inside),
            c_selected_nodes,
        )

        return selected_nodes

    def _mesh2d_count_nodes_in_polygons(
        self, geometry_list: GeometryList, inside: bool
    ) -> int:
        """For internal use only.

        Counts the number of selected mesh node indices.
        This function should be used by clients before `mesh2d_get_nodes_in_polygons`
        for allocating an integer array storing the selection results.

        Returns:
            int: The number of selected nodes
        """
        c_number_of_mesh_nodes = c_int()
        c_geometry_list = CGeometryList.from_geometrylist(geometry_list)

        # Get number of mesh nodes
        self._execute_function(
            self.lib.mkernel_mesh2d_count_nodes_in_polygons,
            self._meshkernelid,
            byref(c_geometry_list),
            c_int(inside),
            byref(c_number_of_mesh_nodes),
        )
        return c_number_of_mesh_nodes.value

    def mesh1d_set(self, mesh1d: Mesh1d) -> None:
        """Sets the one-dimensional mesh state of the MeshKernel.

        Please note that this involves a copy of the data.

        Args:
            mesh1d (Mesh1d): The input data used for setting the state.
        """

        c_mesh1d = CMesh1d.from_mesh1d(mesh1d)

        self._execute_function(
            self.lib.mkernel_mesh1d_set, self._meshkernelid, byref(c_mesh1d)
        )

    def mesh1d_add(self, mesh1d: Mesh1d) -> None:
        """Adds a one-dimensional mesh.

        Please note that this involves a copy of the data.

        Args:
            mesh1d (Mesh1d): The input data used for setting the state.
        """

        c_mesh1d = CMesh1d.from_mesh1d(mesh1d)

        self._execute_function(
            self.lib.mkernel_mesh1d_add, self._meshkernelid, byref(c_mesh1d)
        )

    def mesh1d_get(self) -> Mesh1d:
        """Gets the one-dimensional mesh state from the MeshKernel.

        Please note that this involves a copy of the data.

        Returns:
            Mesh1d: A copy of the two-dimensional mesh state.
        """

        c_mesh1d = self._mesh1d_get_dimensions()

        mesh1d = c_mesh1d.allocate_memory()
        self._execute_function(
            self.lib.mkernel_mesh1d_get_data, self._meshkernelid, byref(c_mesh1d)
        )

        mesh1d.remove_invalid_values(float_invalid_value=self._float_invalid_value)

        return mesh1d

    def _mesh1d_get_dimensions(self) -> CMesh1d:
        """For internal use only.

        Gets the Mesh1d dimensions.
        The integer parameters of the Mesh1D struct are set to the corresponding dimensions.
        The pointers must be set to correctly sized memory before passing the struct to `mesh1d_get`.

        Returns:
            CMesh1d: The CMesh1d with the set dimensions.
        """
        c_mesh1d = CMesh1d()
        self._execute_function(
            self.lib.mkernel_mesh1d_get_dimensions, self._meshkernelid, byref(c_mesh1d)
        )
        return c_mesh1d

    def contacts_set(self, contacts: Contacts) -> None:
        """Sets the contacts.

        Please note that this involves a copy of the data.

        Args:
            contacts (Contacts): The input data used for setting the contacts.
        """

        c_contacts = CContacts.from_contacts(contacts)

        self._execute_function(
            self.lib.mkernel_contacts_set, self._meshkernelid, byref(c_contacts)
        )

    def _contacts_get_dimensions(self) -> CContacts:
        """For internal use only.

        Gets the Contacts dimensions.
        The integer parameters of the Contacts struct are set to the corresponding dimensions.
        The pointers must be set to correctly sized memory before passing the struct to `contacts_get`.

        Returns:
            CContacts: The Contacts with the set dimensions.
        """
        c_contacts = CContacts()

        self._execute_function(
            self.lib.mkernel_contacts_get_dimensions,
            self._meshkernelid,
            byref(c_contacts),
        )

        return c_contacts

    def contacts_get(self) -> Contacts:
        """Gets the Contacts between the Mesh1d and Mesh2d from the MeshKernel.

        Please note that this involves a copy of the data.

        Returns:
            Contacts: The contacts.
        """
        c_contacts = self._contacts_get_dimensions()

        contacts = c_contacts.allocate_memory()

        self._execute_function(
            self.lib.mkernel_contacts_get_data, self._meshkernelid, byref(c_contacts)
        )

        contacts.remove_invalid_values(int_invalid_value=self._int_invalid_value)

        return contacts

    def contacts_compute_single(
        self, node_mask: ndarray, polygons: GeometryList, projection_factor: float
    ) -> None:
        """Computes Mesh1d-Mesh2d contacts, where each single Mesh1d node is connected to one Mesh2d face circumcenter.
        The boundary nodes of Mesh1d (those sharing only one Mesh1d edge) are not connected to any Mesh2d face.

        Args:
            node_mask (ndarray): A boolean array describing whether Mesh1d nodes should or
                                 should not be connected
            polygons (GeometryList): The polygons selecting the area where the contacts will be be generated.
            projection_factor (float): The projection factor used for generating the contacts when 1d nodes are
            not inside the 2d mesh.
        """

        node_mask_int = to_contiguous_numpy_array(node_mask.astype(np.int32))
        c_node_mask = as_ctypes(node_mask_int)
        c_polygons = CGeometryList.from_geometrylist(polygons)

        self._execute_function(
            self.lib.mkernel_contacts_compute_single,
            self._meshkernelid,
            c_node_mask,
            byref(c_polygons),
            c_double(projection_factor),
        )

    def contacts_compute_multiple(self, node_mask: ndarray) -> None:
        """Computes Mesh1d-Mesh2d contacts, where a single Mesh1d node is connected to
        multiple Mesh2d face circumcenters.

        Args:
            node_mask (ndarray): A boolean array describing whether Mesh1d nodes should or
                                 should not be connected
        """

        node_mask_int = to_contiguous_numpy_array(node_mask.astype(np.int32))
        c_node_mask = as_ctypes(node_mask_int)

        self._execute_function(
            self.lib.mkernel_contacts_compute_multiple,
            self._meshkernelid,
            c_node_mask,
        )

    def contacts_compute_with_polygons(
        self, node_mask: ndarray, polygons: GeometryList
    ) -> None:
        """Computes Mesh1d-Mesh2d contacts, where a Mesh2d face per polygon is connected to the closest Mesh1d node.

        Args:
            node_mask (ndarray): A boolean array describing whether Mesh1d nodes should or
                                 should not be connected
            polygons (GeometryList): The polygons in which the closest Mesh2d face to a Mesh1d node will be connected.

        """

        node_mask_int = to_contiguous_numpy_array(node_mask.astype(np.int32))
        c_node_mask = as_ctypes(node_mask_int)
        c_polygons = CGeometryList.from_geometrylist(polygons)

        self._execute_function(
            self.lib.mkernel_contacts_compute_with_polygons,
            self._meshkernelid,
            c_node_mask,
            byref(c_polygons),
        )

    def contacts_compute_with_points(
        self, node_mask: ndarray, polygons: GeometryList = GeometryList()
    ) -> None:
        """Computes Mesh1d-Mesh2d contacts, where Mesh1d nodes are connected to the Mesh2d face mass centers containing
        the input point.

        Args:
            node_mask (ndarray): A boolean array describing whether Mesh1d nodes should or
                                 should not be connected
            polygons (GeometryList, optional):  The polygon selecting the Mesh2d faces to connect.

        """
        node_mask_int = to_contiguous_numpy_array(node_mask.astype(np.int32))
        c_node_mask = as_ctypes(node_mask_int)
        c_polygons = CGeometryList.from_geometrylist(polygons)

        self._execute_function(
            self.lib.mkernel_contacts_compute_with_points,
            self._meshkernelid,
            c_node_mask,
            byref(c_polygons),
        )

    def contacts_compute_boundary(
        self,
        node_mask: ndarray,
        search_radius: float,
        polygons: GeometryList = GeometryList(),
    ) -> None:
        """Computes Mesh1d-Mesh2d contacts, where Mesh1d nodes are connected to the closest Mesh2d faces at the boundary

        Args:
            node_mask (ndarray): A boolean array describing whether Mesh1d nodes should or
                                 should not be connected (1 = generate a connection, 0 = do not generate a connection)
            search_radius (float): The radius used for searching neighboring Mesh2d faces, in meters for cartesian
                                   projection and degrees for spherical projection. If it is equal to the missing value
                                   double, the search radius will be calculated internally.
            polygons (GeometryList, optional): The polygon selecting the Mesh2d faces to connect.

        """

        node_mask_int = to_contiguous_numpy_array(node_mask.astype(np.int32))
        c_node_mask = as_ctypes(node_mask_int)
        c_polygons = CGeometryList.from_geometrylist(polygons)

        self._execute_function(
            self.lib.mkernel_contacts_compute_boundary,
            self._meshkernelid,
            c_node_mask,
            byref(c_polygons),
            c_double(search_radius),
        )

    def mesh2d_casulli_derefinement(self) -> None:
        """
        De-refine the whole mesh using the Casulli algorithm
        """
        self._execute_function(
            self.lib.mkernel_mesh2d_casulli_derefinement,
            self._meshkernelid,
        )

    def mesh2d_casulli_derefinement_on_polygon(
        self,
        polygon: GeometryList,
    ) -> None:
        """
        De-refine a mesh region using the Casulli algorithm

        Args:
            polygon (GeometryList): The input polygon.
        """
        c_polygon = CGeometryList.from_geometrylist(polygon)
        self._execute_function(
            self.lib.mkernel_mesh2d_casulli_derefinement_on_polygon,
            self._meshkernelid,
            byref(c_polygon),
        )

    def mesh2d_casulli_refinement(self) -> None:
        """
        Refine the whole mesh using the Casulli algorithm
        """
        self._execute_function(
            self.lib.mkernel_mesh2d_casulli_refinement,
            self._meshkernelid,
        )

    def mesh2d_casulli_refinement_on_polygon(
        self,
        polygon: GeometryList,
    ) -> None:
        """
        Refine a mesh region using the Casulli algorithm

        Args:
            polygon (GeometryList): The input polygon.
        """
        c_polygon = CGeometryList.from_geometrylist(polygon)
        self._execute_function(
            self.lib.mkernel_mesh2d_casulli_refinement_on_polygon,
            self._meshkernelid,
            byref(c_polygon),
        )

    def mkernel_delete_property(self, propertyId: int):
        """
        Deletes a property and its calculator.

        Args:
            propertyId (int): The property id.
        """

        self._execute_function(
            self.lib.mkernel_deallocate_property,
            c_int(propertyId),
        )

    def mkernel_set_property(
        self,
        projection: ProjectionType,
        interpolationType: InterpolationType,
        sampleData: GeometryList,
    ) -> int:
        """
        Sets the property data for the mesh, the sample data points do not have to match the mesh2d nodes.

        Args:
            projection (ProjectionType): The projection type used by the sample data.
            interpolationType (InterpolationType): The type of interpolation required, triangulation or averaging.
            sampleData (GeometryList): The sample data and associated sample data points

        Returns:
            int: The id of the property.
        """

        c_sampleData = CGeometryList.from_geometrylist(sampleData)
        propertyId = c_int()

        self._execute_function(
            self.lib.mkernel_mesh2d_set_property,
            c_int(projection.value),
            interpolationType,
            byref(c_sampleData),
            byref(propertyId),
        )

        return propertyId.value

    def mkernel_mesh2d_casulli_refinement_based_on_depths(
        self,
        polygons: GeometryList,
        propertyId: int,
        meshRefinementParameters: MeshRefinementParameters,
        minimumRefinementDepth: float,
    ) -> None:
        """
        Refine mesh using the Casulli refinement algorithm based on the depth values.

        Args:
            polygons (GeometryList): The polygon within which the refinement is computed.
            propertyId (int): The identifier of the property and its associated interpolator.
            meshRefinementParameters (MeshRefinementParameters): Parameters indicating how the mesh is to be refined.
            minimumRefinementDepth (float): Nodes with depth value less than this value will not be marked for refinement.
        """

        c_polygons = CGeometryList.from_geometrylist(polygons)
        c_refinement_params = CMeshRefinementParameters.from_meshrefinementparameters(
            meshRefinementParameters
        )
        self._execute_function(
            self.lib.mkernel_mesh2d_casulli_refinement_wrt_depths,
            self._meshkernelid,
            byref(c_polygons),
            c_int(propertyId),
            byref(c_refinement_params),
            c_double(minimumRefinementDepth),
        )

    def mesh2d_compute_orthogonalization(
        self,
        project_to_land_boundary_option: ProjectToLandBoundaryOption,
        orthogonalization_parameters: OrthogonalizationParameters,
        land_boundaries: GeometryList,
        selecting_polygon: GeometryList = GeometryList(),
    ) -> None:
        """Orthogonalizes the Mesh2d.
        The function modifies the mesh for achieving orthogonality between the edges
        and the segments connecting the face circumcenters.
        The amount of orthogonality is traded against the mesh smoothing (in this case the equality of face areas).

        Args:
            project_to_land_boundary_option (ProjectToLandBoundaryOption): The option to determine how to snap to
                                                                           land boundaries.
            orthogonalization_parameters (OrthogonalizationParameters): The orthogonalization parameters.
            land_boundaries (GeometryList): The land boundaries to account for in the orthogonalization process.
            selecting_polygon (GeometryList, optional): The polygon where to perform the orthogonalization.
        """

        c_orthogonalization_params = (
            COrthogonalizationParameters.from_orthogonalizationparameters(
                orthogonalization_parameters
            )
        )
        c_selecting_polygon = CGeometryList.from_geometrylist(selecting_polygon)
        c_land_boundaries = CGeometryList.from_geometrylist(land_boundaries)

        self._execute_function(
            self.lib.mkernel_mesh2d_compute_orthogonalization,
            self._meshkernelid,
            c_int(project_to_land_boundary_option),
            byref(c_orthogonalization_params),
            byref(c_selecting_polygon),
            byref(c_land_boundaries),
        )

    def mesh2d_get_orthogonality(self) -> GeometryList:
        """Gets the mesh orthogonality, expressed as the ratio between the edges and
        the segments connecting the face circumcenters.

        Returns:
            GeometryList: The geometry list with the orthogonality values of each edge.
        """

        number_of_coordinates = self._mesh2d_get_dimensions().num_edges

        x_coordinates = np.empty(number_of_coordinates, dtype=np.double)
        y_coordinates = np.empty(number_of_coordinates, dtype=np.double)
        values = np.empty(number_of_coordinates, dtype=np.double)
        geometry_list_out = GeometryList(x_coordinates, y_coordinates, values)

        c_geometry_list_out = CGeometryList.from_geometrylist(geometry_list_out)
        self._execute_function(
            self.lib.mkernel_mesh2d_get_orthogonality,
            self._meshkernelid,
            byref(c_geometry_list_out),
        )

        self.__map_to_valid_values(geometry_list_out, Mesh2dLocation.EDGES)

        return geometry_list_out

    def mesh2d_get_property(
        self, mesh2dLocation: Mesh2dLocation, property: Mesh2d.Property
    ) -> GeometryList:
        """Gets the polygons matching the metric value within the minimum and maximum value.

        Args:

            property (Mesh2d.Property): The property to retrieve

        Returns:
            GeometryList: The resulting geometry list containing the value of the properties
        """

        c_geometry_list_dimension = c_int()

        self._execute_function(
            self.lib.mkernel_mesh2d_get_property_dimension,
            self._meshkernelid,
            c_int(property),
            byref(c_geometry_list_dimension),
        )

        n_coordinates = c_geometry_list_dimension.value
        x_coordinates = np.empty(n_coordinates, dtype=np.double)
        y_coordinates = np.empty(n_coordinates, dtype=np.double)
        values = np.empty(n_coordinates, dtype=np.double)

        property_list = GeometryList(
            x_coordinates=x_coordinates, y_coordinates=y_coordinates, values=values
        )
        c_property_list = CGeometryList.from_geometrylist(property_list)

        self._execute_function(
            self.lib.mkernel_mesh2d_get_property,
            self._meshkernelid,
            c_int(property),
            c_int(mesh2dLocation.value),
            byref(c_property_list),
        )

        return property_list

    def mesh2d_get_smoothness(self):
        """Gets the smoothness, expressed as the ratio between the values of two neighboring faces areas.

        Returns:
            GeometryList: The geometry list with the smoothness values of each edge.
        """

        number_of_coordinates = self._mesh2d_get_dimensions().num_edges

        x_coordinates = np.empty(number_of_coordinates, dtype=np.double)
        y_coordinates = np.empty(number_of_coordinates, dtype=np.double)
        values = np.empty(number_of_coordinates, dtype=np.double)
        geometry_list_out = GeometryList(x_coordinates, y_coordinates, values)

        c_geometry_list_out = CGeometryList.from_geometrylist(geometry_list_out)
        self._execute_function(
            self.lib.mkernel_mesh2d_get_smoothness,
            self._meshkernelid,
            byref(c_geometry_list_out),
        )

        self.__map_to_valid_values(geometry_list_out, Mesh2dLocation.EDGES)

        return geometry_list_out

    def mesh2d_connect_meshes(self, mesh2d: Mesh2d, search_fraction: float) -> None:
        """Connect a mesh to an existing mesh

        Args:
            mesh2d (Mesh2d): The mesh to connect to the existing mesh
            search_fraction (float): Fraction of the shortest edge (along an edge to be connected)
                                     to use when determining neighbour edge closeness
        """

        c_mesh2d = CMesh2d.from_mesh2d(mesh2d)

        self._execute_function(
            self.lib.mkernel_mesh2d_connect_meshes,
            self._meshkernelid,
            byref(c_mesh2d),
            c_double(search_fraction),
        )

    def _get_error(self) -> str:
        c_string_size = 512
        c_error_message = create_string_buffer(c_string_size)
        self.lib.mkernel_get_error(c_error_message)
        return c_error_message.value.decode("ASCII")

    def _get_geometry_error(self) -> Tuple[int, Mesh2dLocation]:
        """Get geometry error information
        Returns:
            index (int): The index
            location (int): The location
        """
        index = c_int()
        location = c_int()
        self.lib.mkernel_get_geometry_error(byref(index), byref(location))
        return index.value, Mesh2dLocation(location)

    def mesh2d_triangulation_interpolation(
        self,
        samples: GeometryList,
        location_type: Mesh2dLocation,
    ) -> GeometryList:
        """Performs triangulation interpolation of samples.

        Args:
            samples (GeometryList): The samples to interpolate.
            location_type (Mesh2dLocation): The location type on which to interpolate.

        Returns:
            GeometryList: The interpolated samples.
        """
        c_samples = CGeometryList.from_geometrylist(samples)

        number_of_coordinates = self._get_num_coordinates(location_type)
        x_coordinates = np.empty(number_of_coordinates, dtype=np.double)
        y_coordinates = np.empty(number_of_coordinates, dtype=np.double)
        values = np.empty(number_of_coordinates, dtype=np.double)
        interpolated_samples = GeometryList(x_coordinates, y_coordinates, values)

        c_interpolated_samples = CGeometryList.from_geometrylist(interpolated_samples)

        self._execute_function(
            self.lib.mkernel_mesh2d_triangulation_interpolation,
            self._meshkernelid,
            byref(c_samples),
            c_int(location_type),
            byref(c_interpolated_samples),
        )

        return interpolated_samples

    def mesh2d_averaging_interpolation(
        self,
        samples: GeometryList,
        location_type: Mesh2dLocation,
        averaging_method: AveragingMethod,
        relative_search_size: float,
        min_samples: int,
    ) -> GeometryList:
        """Performs averaging interpolation of samples.

        Args:
            samples (GeometryList): The samples to interpolate.
            location_type (Mesh2dLocation): The location type on which to interpolate.
            averaging_method (AveragingMethod): The averaging method.
            relative_search_size (float): The relative search size.
            min_samples (int): The minimum number of samples used for some interpolation algorithms to perform
                               a valid interpolation.

        Returns:
            GeometryList: The interpolated samples.
        """
        c_samples = CGeometryList.from_geometrylist(samples)

        number_of_coordinates = self._get_num_coordinates(location_type)
        x_coordinates = np.empty(number_of_coordinates, dtype=np.double)
        y_coordinates = np.empty(number_of_coordinates, dtype=np.double)
        values = np.empty(number_of_coordinates, dtype=np.double)
        interpolated_samples = GeometryList(x_coordinates, y_coordinates, values)

        c_interpolated_samples = CGeometryList.from_geometrylist(interpolated_samples)

        self._execute_function(
            self.lib.mkernel_mesh2d_averaging_interpolation,
            self._meshkernelid,
            byref(c_samples),
            c_int(location_type),
            c_int(averaging_method),
            c_double(relative_search_size),
            c_size_t(min_samples),
            byref(c_interpolated_samples),
        )

        return interpolated_samples

    def mesh2d_convert_projection(
        self,
        projection: ProjectionType,
        zone: str,
    ) -> None:
        """Converts the projection of a mesh2d.

        Args:
            projection (ProjectionType): The new mesh projection.
            zone (str): The UTM zone and information string.
        """
        self._execute_function(
            self.lib.mkernel_mesh2d_convert_projection,
            self._meshkernelid,
            c_int(projection),
            c_char_p(zone.encode()),
        )

    def get_projection(self) -> ProjectionType:
        """Gets the projection type of the meshkernel state

        Returns:
                   ProjectionType: The projection type
        """
        projection = c_int()
        self._execute_function(
            self.lib.mkernel_get_projection, self._meshkernelid, byref(projection)
        )

        return ProjectionType(projection.value)

    def get_meshkernel_version(self) -> str:
        """Get the version of the underlying C++ MeshKernel library

        Returns:
            str: The version string
        """

        c_string_size = 64
        c_meshkernel_version = create_string_buffer(c_string_size)
        self.lib.mkernel_get_version(c_meshkernel_version)
        return c_meshkernel_version.value.decode("ASCII")

    def get_meshkernelpy_version(self) -> str:
        """Get the version of this Python wrapper

        Returns:
            str: The version string
        """

        return __version__

    def mkernel_get_separator(self) -> float:
        """Get the separator used in MeshKernel

        Returns:
            float: The separator
        """
        self.lib.mkernel_get_separator.restype = c_double

        return self.lib.mkernel_get_separator()

    def mkernel_get_inner_outer_separator(self) -> float:
        """Get the separator used in MeshKernel for separating the outer perimeter of a polygon from is inner perimeter

        Returns:
            float: The polygon inner/outer separator
        """

        self.lib.mkernel_get_inner_outer_separator.restype = c_double

        return self.lib.mkernel_get_inner_outer_separator()

    def _execute_function(self, function, *args):
        """Utility function to execute a C function of MeshKernel and checks its status.

        Args:
            function (Callable): The function which we want to call.
            args: Arguments which will be passed to `function`.

        Raises:
            MeshKernelError: This exception gets raised,
                             if the MeshKernel library reports an error.
        """
        exit_code = function(*args)
        if exit_code != self._exit_code.SUCCESS:
            error_message = self._get_error()
            if exit_code == self._exit_code.MESHKERNEL_ERROR:
                raise MeshKernelError("MeshKernelError", error_message)
            elif exit_code == self._exit_code.NOT_IMPLEMENTED_ERROR:
                raise MeshKernelError("NotImplementedError", error_message)
            elif exit_code == self._exit_code.ALGORITHM_ERROR:
                raise MeshKernelError("AlgorithmError", error_message)
            elif exit_code == self._exit_code.CONSTRAINT_ERROR:
                raise MeshKernelError("ConstraintError", error_message)
            elif exit_code == self._exit_code.MESH_GEOMETRY_ERROR:
                raise MeshGeometryError(error_message, self._get_geometry_error())
            elif exit_code == self._exit_code.LINEAR_ALGEBRA_ERROR:
                raise MeshKernelError("LinearAlgebraError", error_message)
            elif exit_code == self._exit_code.RANGE_ERROR:
                raise MeshKernelError("RangeError", error_message)
            elif exit_code == self._exit_code.STDLIB_EXCEPTION:
                raise MeshKernelError("STDLibException", error_message)
            elif exit_code == self._exit_code.UNKNOWN_EXCEPTION:
                raise MeshKernelError("UnknownException", error_message)

    def _curvilineargrid_get_dimensions(self) -> CCurvilinearGrid:
        """For internal use only.

        Gets the curvilinear grid dimensions.
        The integer parameters of the Curvilinear grid struct are set to the corresponding dimensions.
        The pointers must be set to correctly sized memory before passing the struct to `curvilineargrid_get`.

        Returns:
            CMesh1d: The CCurvilinearGrid with the set dimensions.
        """
        c_curvilineargrid = CCurvilinearGrid()
        self._execute_function(
            self.lib.mkernel_curvilinear_get_dimensions,
            self._meshkernelid,
            byref(c_curvilineargrid),
        )
        return c_curvilineargrid

    def curvilineargrid_get(self) -> CurvilinearGrid:
        """Gets the curvilinear grid state from the MeshKernel.

        Please note that this involves a copy of the data.

        Returns:
            CurvilinearGrid: A copy of the curvilinear grid state.
        """

        c_curvilineargrid = self._curvilineargrid_get_dimensions()

        curvilineargrid = c_curvilineargrid.allocate_memory()
        self._execute_function(
            self.lib.mkernel_curvilinear_get_data,
            self._meshkernelid,
            byref(c_curvilineargrid),
        )

        return curvilineargrid

    def curvilinear_compute_transfinite_from_splines(
        self,
        splines: GeometryList,
        curvilinear_parameters: CurvilinearParameters,
    ) -> None:
        """Generates curvilinear grid from splines with transfinite interpolation.

        Args:
            splines (GeometryList): The spline to use for curvilinear grid generation.
            curvilinear_parameters (CurvilinearParameters): The curvilinear grid parameters.
        """

        c_curvilinear_params = CCurvilinearParameters.from_curvilinearParameters(
            curvilinear_parameters
        )
        c_splines = CGeometryList.from_geometrylist(splines)

        self._execute_function(
            self.lib.mkernel_curvilinear_compute_transfinite_from_splines,
            self._meshkernelid,
            byref(c_splines),
            byref(c_curvilinear_params),
        )

    def curvilinear_compute_orthogonal_from_splines(
        self,
        splines: GeometryList,
        curvilinear_parameters: CurvilinearParameters,
        splines_to_curvilinear_parameters: SplinesToCurvilinearParameters,
    ) -> None:
        """Generates curvilinear grid from splines with the advancing front method.

        Args:
            splines (GeometryList): The spline to use for curvilinear grid generation.
            curvilinear_parameters (CurvilinearParameters): The curvilinear grid parameters.
            splines_to_curvilinear_parameters (SplinesToCurvilinearParameters): Additional parameters required
            by the algorithm.
        """

        c_splines = CGeometryList.from_geometrylist(splines)
        c_curvilinear_params = CCurvilinearParameters.from_curvilinearParameters(
            curvilinear_parameters
        )
        c_splines_to_curvilinear_params = (
            CSplinesToCurvilinearParameters.from_splinesToCurvilinearParameters(
                splines_to_curvilinear_parameters
            )
        )
        self._execute_function(
            self.lib.mkernel_curvilinear_compute_orthogonal_grid_from_splines,
            self._meshkernelid,
            byref(c_splines),
            byref(c_curvilinear_params),
            byref(c_splines_to_curvilinear_params),
        )

    def curvilinear_compute_curvature(
        self,
        direction: CurvilinearDirection,
    ) -> ndarray:
        """Computes the curvature of a curvilinear grid.

        Args:
            direction (int): The direction in which to compute the smoothness.
        """

        curvilinear_grid_dimensions = self._curvilineargrid_get_dimensions()

        num_m = curvilinear_grid_dimensions.num_m
        num_n = curvilinear_grid_dimensions.num_n

        result = np.empty(num_m * num_n, dtype=np.double)
        c_result = np.ctypeslib.as_ctypes(result)
        self._execute_function(
            self.lib.mkernel_curvilinear_compute_curvature,
            self._meshkernelid,
            c_int(direction),
            byref(c_result),
        )
        return result

    def curvilinear_compute_smoothness(
        self,
        direction: CurvilinearDirection,
    ) -> ndarray:
        """Computes the smoothness of a curvilinear grid.

        Args:
            direction (int): The direction in which to compute the smoothness.
        """

        curvilinear_grid_dimensions = self._curvilineargrid_get_dimensions()

        num_m = curvilinear_grid_dimensions.num_m
        num_n = curvilinear_grid_dimensions.num_n

        result = np.empty(num_m * num_n, dtype=np.double)
        c_result = np.ctypeslib.as_ctypes(result)
        self._execute_function(
            self.lib.mkernel_curvilinear_compute_smoothness,
            self._meshkernelid,
            c_int(direction),
            byref(c_result),
        )
        return result

    def curvilinear_convert_to_mesh2d(self) -> None:
        """Converts a curvilinear grid to an unstructured mesh"""
        self._execute_function(
            self.lib.mkernel_curvilinear_convert_to_mesh2d, self._meshkernelid
        )

    def curvilinear_compute_rectangular_grid(
        self, make_grid_parameters: MakeGridParameters
    ) -> None:
        """Makes a rectangular grid

        Args:
            make_grid_parameters (MakeGridParameters): The parameters used for making the uniform grid
        """

        c_make_grid_parameters = CMakeGridParameters.from_makegridparameters(
            make_grid_parameters
        )

        self._execute_function(
            self.lib.mkernel_curvilinear_compute_rectangular_grid,
            self._meshkernelid,
            byref(c_make_grid_parameters),
        )

    def curvilinear_compute_rectangular_grid_from_polygon(
        self,
        make_grid_parameters: MakeGridParameters,
        geometry_list: GeometryList,
    ) -> None:
        """Makes a rectangular grid from polygons.

        Args:
            make_grid_parameters (MakeGridParameters): The parameters used for making the uniform grid
            geometry_list (GeometryList): The polygon within which the grid will be generated
        """

        c_make_grid_parameters = CMakeGridParameters.from_makegridparameters(
            make_grid_parameters
        )

        c_geometry_list = CGeometryList.from_geometrylist(geometry_list)

        self._execute_function(
            self.lib.mkernel_curvilinear_compute_rectangular_grid_from_polygon,
            self._meshkernelid,
            byref(c_make_grid_parameters),
            byref(c_geometry_list),
        )

    def curvilinear_compute_rectangular_grid_on_extension(
        self,
        make_grid_parameters: MakeGridParameters,
    ) -> None:
        """Makes a rectangular grid on defined extension.

        Args:
            make_grid_parameters (MakeGridParameters): The parameters used for making the uniform grid
        """

        c_make_grid_parameters = CMakeGridParameters.from_makegridparameters(
            make_grid_parameters
        )

        self._execute_function(
            self.lib.mkernel_curvilinear_compute_rectangular_grid_on_extension,
            self._meshkernelid,
            byref(c_make_grid_parameters),
        )

    def curvilinear_refine(
        self,
        x_lower_left_corner: float,
        y_lower_left_corner: float,
        x_upper_right_corner: float,
        y_upper_right_corner: float,
        refinement: int,
    ) -> None:
        """Directional curvilinear grid refinement.
        Additional gridlines are added perpendicularly to the segment defined by lower_left_corner
        and upper_right_corner

        Args:
            x_lower_left_corner (float): The x coordinate of the lower left corner of the block to refine.
            y_lower_left_corner (float): The y coordinate of the lower left corner of the block to refine.
            x_upper_right_corner (float): The x coordinate of the upper right corner of the block to refine.
            y_upper_right_corner (float): The y coordinate of the upper right corner of the block to refine.
            refinement (int): The number of grid lines to add between lower left and upper right
        """
        self._execute_function(
            self.lib.mkernel_curvilinear_refine,
            self._meshkernelid,
            c_double(x_lower_left_corner),
            c_double(y_lower_left_corner),
            c_double(x_upper_right_corner),
            c_double(y_upper_right_corner),
            c_int(refinement),
        )

    def curvilinear_derefine(
        self,
        x_lower_left_corner: float,
        y_lower_left_corner: float,
        x_upper_right_corner: float,
        y_upper_right_corner: float,
    ) -> None:
        """Directional curvilinear grid derefinement.
        Additional gridlines are removed perpendicularly to the segment defined by lower_left_corner
        and upper_right_corner

        Args:
            x_lower_left_corner (float): The x coordinate of the lower left corner of the block to refine.
            y_lower_left_corner (float): The y coordinate of the lower left corner of the block to refine.
            x_upper_right_corner (float): The x coordinate of the upper right corner of the block to refine.
            y_upper_right_corner (float): The y coordinate of the upper right corner of the block to refine.
        """
        self._execute_function(
            self.lib.mkernel_curvilinear_derefine,
            self._meshkernelid,
            c_double(x_lower_left_corner),
            c_double(y_lower_left_corner),
            c_double(x_upper_right_corner),
            c_double(y_upper_right_corner),
        )

    def curvilinear_compute_transfinite_from_polygon(
        self,
        geometry_list: GeometryList,
        first_node: int,
        second_node: int,
        third_node: int,
        use_fourth_side: bool,
    ) -> None:
        """Computes a curvilinear mesh in a polygon. 3 separate polygon nodes need to be selected.

        Args:
            geometry_list (GeometryList): The input polygon.
            first_node (int): The first selected node.
            second_node (int): The second selected node.
            third_node (int): The third selected node.
            use_fourth_side (bool): Use the fourth polygon side to compute the curvilinear grid.
        """

        c_geometry_list = CGeometryList.from_geometrylist(geometry_list)
        use_fourth_side_int = 1 if use_fourth_side else 0

        self._execute_function(
            self.lib.mkernel_curvilinear_compute_transfinite_from_polygon,
            self._meshkernelid,
            byref(c_geometry_list),
            c_int(first_node),
            c_int(second_node),
            c_int(third_node),
            c_int(use_fourth_side_int),
        )

    def curvilinear_compute_transfinite_from_triangle(
        self,
        geometry_list: GeometryList,
        first_node: int,
        second_node: int,
        third_node: int,
    ) -> None:
        """Computes a curvilinear mesh in a triangle. 3 separate polygon nodes need to be selected.

        Args:
            geometry_list (GeometryList): The input polygon.
            first_node (int): The first selected node.
            second_node (int): The second selected node.
            third_node (int): The third selected node.
        """

        c_geometry_list = CGeometryList.from_geometrylist(geometry_list)

        self._execute_function(
            self.lib.mkernel_curvilinear_compute_transfinite_from_polygon,
            self._meshkernelid,
            byref(c_geometry_list),
            c_int(first_node),
            c_int(second_node),
            c_int(third_node),
        )

    def curvilinear_initialize_orthogonalize(
        self, orthogonalization_parameters: OrthogonalizationParameters
    ) -> None:
        """Initializes the algorithm for performing curvilinear grid orthogonalization.

        Args:
            orthogonalization_parameters (OrthogonalizationParameters): The input orthogonalization parameters.
        """

        c_orthogonalization_parameters = (
            COrthogonalizationParameters.from_orthogonalizationparameters(
                orthogonalization_parameters
            )
        )

        self._execute_function(
            self.lib.mkernel_curvilinear_initialize_orthogonalize,
            self._meshkernelid,
            byref(c_orthogonalization_parameters),
        )

    def curvilinear_set_block_orthogonalize(
        self,
        x_lower_left_corner: float,
        y_lower_left_corner: float,
        x_upper_right_corner: float,
        y_upper_right_corner: float,
    ) -> None:
        """Defines a block on the curvilinear grid where to perform orthogonalization

        Args:
            x_lower_left_corner (float): The x coordinate of the lower left corner of the block to orthogonalize.
            y_lower_left_corner (float): The y coordinate of the lower left corner of the block to orthogonalize.
            x_upper_right_corner (float): The x coordinate of the upper right corner of the block to orthogonalize.
            y_upper_right_corner (float): The y coordinate of the upper right corner of the block to orthogonalize.
        """

        self._execute_function(
            self.lib.mkernel_curvilinear_set_block_orthogonalize,
            self._meshkernelid,
            c_double(x_lower_left_corner),
            c_double(y_lower_left_corner),
            c_double(x_upper_right_corner),
            c_double(y_upper_right_corner),
        )

    def curvilinear_set_frozen_lines_orthogonalize(
        self,
        x_first_gridline_node: float,
        y_first_gridline_node: float,
        x_second_gridline_node: float,
        y_second_gridline_node: float,
    ) -> None:
        """Freezes a curvilinear grid line during the orthogonalization process

        Args:
            x_first_gridline_node (float): The x coordinate of the first point of the line to freeze.
            y_first_gridline_node (float): The y coordinate of the first point of the line to freeze.
            x_second_gridline_node (float): The x coordinate of the second point of the line to freeze.
            y_second_gridline_node (float): The y coordinate of the second point of the line to freeze.
        """

        self._execute_function(
            self.lib.mkernel_curvilinear_set_frozen_lines_orthogonalize,
            self._meshkernelid,
            c_double(x_first_gridline_node),
            c_double(y_first_gridline_node),
            c_double(x_second_gridline_node),
            c_double(y_second_gridline_node),
        )

    def curvilinear_orthogonalize(self) -> None:
        """Performs curvilinear grid orthogonalization and finalizes the algorithm"""
        self._execute_function(
            self.lib.mkernel_curvilinear_orthogonalize, self._meshkernelid
        )
        self._execute_function(
            self.lib.mkernel_curvilinear_finalize_orthogonalize, self._meshkernelid
        )

    def curvilinear_smoothing(
        self,
        smoothing_iterations: int,
        x_lower_left_corner: float,
        y_lower_left_corner: float,
        x_upper_right_corner: float,
        y_upper_right_corner: float,
    ) -> None:
        """Smooths a curvilinear grid.

        Args:
            smoothing_iterations (int): The number of smoothing iterations to perform
            x_lower_left_corner (float): The x coordinate of the lower left corner of the block to refine.
            y_lower_left_corner (float): The y coordinate of the lower left corner of the block to refine.
            x_upper_right_corner (float): The x coordinate of the upper right corner of the block to refine.
            y_upper_right_corner (float): The y coordinate of the upper right corner of the block to refine.
        """

        self._execute_function(
            self.lib.mkernel_curvilinear_smoothing,
            self._meshkernelid,
            c_int(smoothing_iterations),
            c_double(x_lower_left_corner),
            c_double(y_lower_left_corner),
            c_double(x_upper_right_corner),
            c_double(y_upper_right_corner),
        )

    def curvilinear_smoothing_directional(
        self,
        smoothing_iterations: int,
        x_first_grid_line_node: float,
        y_first_grid_line_node: float,
        x_second_grid_line_node: float,
        y_second_grid_line_node: float,
        x_lower_left_corner: float,
        y_lower_left_corner: float,
        x_upper_right_corner: float,
        y_upper_right_corner: float,
    ) -> None:
        """Smooths a curvilinear grid along the direction specified by a segment.

        Args:
            smoothing_iterations (int): The number of smoothing iterations to perform
            x_first_grid_line_node (float): The x coordinate of the first curvilinear grid node.
            y_first_grid_line_node (float): The y coordinate of the first curvilinear grid node.
            x_second_grid_line_node (float): The x coordinate of the second curvilinear grid node.
            y_second_grid_line_node (float): The y coordinate of the second curvilinear grid node.
            x_lower_left_corner (float): The x coordinate of the lower left corner of the block to refine.
            y_lower_left_corner (float): The y coordinate of the lower left corner of the block to refine.
            x_upper_right_corner (float): The x coordinate of the upper right corner of the block to refine.
            y_upper_right_corner (float): The y coordinate of the upper right corner of the block to refine.
        """

        self._execute_function(
            self.lib.mkernel_curvilinear_smoothing_directional,
            self._meshkernelid,
            c_int(smoothing_iterations),
            c_double(x_first_grid_line_node),
            c_double(y_first_grid_line_node),
            c_double(x_second_grid_line_node),
            c_double(y_second_grid_line_node),
            c_double(x_lower_left_corner),
            c_double(y_lower_left_corner),
            c_double(x_upper_right_corner),
            c_double(y_upper_right_corner),
        )

    def curvilinear_initialize_line_shift(self) -> None:
        """Initializes the curvilinear line shift algorithm"""
        self._execute_function(
            self.lib.mkernel_curvilinear_initialize_line_shift, self._meshkernelid
        )

    def curvilinear_set_line_line_shift(
        self,
        x_first_grid_line_node: float,
        y_first_grid_line_node: float,
        x_second_grid_line_node: float,
        y_second_grid_line_node: float,
    ) -> None:
        """Sets the start and end grid nodes of the grid line to shift.

        Args:
            x_first_grid_line_node (float): The x coordinate of the first curvilinear grid node to shift.
            y_first_grid_line_node (float): The y coordinate of the first curvilinear grid node to shift.
            x_second_grid_line_node (float): The x coordinate of the second curvilinear grid node to shift.
            y_second_grid_line_node (float): The y coordinate of the second curvilinear grid node to shift.
        """

        self._execute_function(
            self.lib.mkernel_curvilinear_set_line_line_shift,
            self._meshkernelid,
            c_double(x_first_grid_line_node),
            c_double(y_first_grid_line_node),
            c_double(x_second_grid_line_node),
            c_double(y_second_grid_line_node),
        )

    def curvilinear_set_block_line_shift(
        self,
        x_lower_left_corner: float,
        y_lower_left_corner: float,
        x_upper_right_corner: float,
        y_upper_right_corner: float,
    ) -> None:
        """Defines a block on the curvilinear where the grid line shifting is distributed.

        Args:
            x_lower_left_corner (float): The x coordinate of the lower left corner of the block.
            y_lower_left_corner (float): The y coordinate of the lower left corner of the block.
            x_upper_right_corner (float): The x coordinate of the upper right corner of the block.
            y_upper_right_corner (float): The y coordinate of the upper right corner of the block.
        """

        self._execute_function(
            self.lib.mkernel_curvilinear_set_block_line_shift,
            self._meshkernelid,
            c_double(x_lower_left_corner),
            c_double(y_lower_left_corner),
            c_double(x_upper_right_corner),
            c_double(y_upper_right_corner),
        )

    def curvilinear_move_node_line_shift(
        self,
        x_from_coordinate: float,
        y_from_coordinate: float,
        x_to_coordinate: float,
        y_to_coordinate: float,
    ) -> None:
        """Moves a node of the line to shift. The operation can be performed multiple times.

        Args:
            x_from_coordinate (float): The x coordinate of the node to move.
            y_from_coordinate (float): The y coordinate of the node to move.
            x_to_coordinate (float): The x coordinate of the new node position.
            y_to_coordinate (float): The y coordinate of the new node position.
        """

        self._execute_function(
            self.lib.mkernel_curvilinear_move_node_line_shift,
            self._meshkernelid,
            c_double(x_from_coordinate),
            c_double(y_from_coordinate),
            c_double(x_to_coordinate),
            c_double(y_to_coordinate),
        )

    def curvilinear_line_shift(self):
        """Performs the curvilinear line shifting and finalizes the algorithm."""
        self._execute_function(
            self.lib.mkernel_curvilinear_line_shift, self._meshkernelid
        )
        self._execute_function(
            self.lib.mkernel_curvilinear_finalize_line_shift, self._meshkernelid
        )

    def curvilinear_move_node(
        self,
        x_from_point: float,
        y_from_point: float,
        x_to_point: float,
        y_to_point: float,
    ) -> None:
        """Moves a point of a curvilinear grid from one location to another.

        Args:
            x_from_point (float): The x coordinate of point to move.
            y_from_point (float): The y coordinate of point to move.
            x_to_point (float): The new x coordinate of the point.
            y_to_point (float): The new y coordinate of the point.
        """
        self._execute_function(
            self.lib.mkernel_curvilinear_move_node,
            self._meshkernelid,
            c_double(x_from_point),
            c_double(y_from_point),
            c_double(x_to_point),
            c_double(y_to_point),
        )

    def curvilinear_insert_face(self, x_coordinate: float, y_coordinate: float) -> None:
        """Inserts a new face on a curvilinear grid.
        The new face will be inserted on top of the closest edge by linear extrapolation.

        Args:
            x_coordinate (float): The x coordinate of the point used for finding the closest face.
            y_coordinate (float): The y coordinate of the point used for finding the closest face.
        """

        self._execute_function(
            self.lib.mkernel_curvilinear_insert_face,
            self._meshkernelid,
            c_double(x_coordinate),
            c_double(y_coordinate),
        )

    def curvilinear_line_attraction_repulsion(
        self,
        repulsion_parameter: float,
        x_first_grid_line_node: float,
        y_first_grid_line_node: float,
        x_second_grid_line_node: float,
        y_second_grid_line_node: float,
        x_lower_left_corner: float,
        y_lower_left_corner: float,
        x_upper_right_corner: float,
        y_upper_right_corner: float,
    ):
        """Attracts/repulses grid lines in a block towards another set grid line.

        Args:
            repulsion_parameter (float): The repulsion parameter.
            x_first_grid_line_node (float): The x coordinate of the first node.
            y_first_grid_line_node (float): The y coordinate of the first node.
            x_second_grid_line_node (float): The x coordinate of the second node.
            y_second_grid_line_node (float): The y coordinate of the second node.
            x_lower_left_corner (float): The x coordinate of the lower left corner of the block.
            y_lower_left_corner (float): The y coordinate of the lower left corner of the block.
            x_upper_right_corner (float): The x coordinate of the upper right corner of the block.
            y_upper_right_corner (float): The y coordinate of the upper right corner of the block.
        """
        self._execute_function(
            self.lib.mkernel_curvilinear_line_attraction_repulsion,
            self._meshkernelid,
            c_double(repulsion_parameter),
            c_double(x_first_grid_line_node),
            c_double(y_first_grid_line_node),
            c_double(x_second_grid_line_node),
            c_double(y_second_grid_line_node),
            c_double(x_lower_left_corner),
            c_double(y_lower_left_corner),
            c_double(x_upper_right_corner),
            c_double(y_upper_right_corner),
        )

    def curvilinear_line_mirror(
        self,
        mirroring_factor: float,
        x_first_grid_line_node: float,
        y_first_grid_line_node: float,
        x_second_grid_line_node: float,
        y_second_grid_line_node: float,
    ) -> None:
        """Mirrors a boundary grid line outwards. The boundary grid line is defined by its start and end points

        Args:
            mirroring_factor (float): The mirroring factor.
            x_first_grid_line_node (float): The x coordinate of the first node.
            y_first_grid_line_node (float): The y coordinate of the first node.
            x_second_grid_line_node (float): The x coordinate of the second node.
            y_second_grid_line_node (float): The y coordinate of the second node.
        """
        self._execute_function(
            self.lib.mkernel_curvilinear_line_mirror,
            self._meshkernelid,
            c_double(mirroring_factor),
            c_double(x_first_grid_line_node),
            c_double(y_first_grid_line_node),
            c_double(x_second_grid_line_node),
            c_double(y_second_grid_line_node),
        )

    def curvilinear_delete_node(self, x_coordinate: float, y_coordinate: float) -> None:
        """Delete the node closest to a point

        Args:
            x_coordinate (float): The x coordinate of the point.
            y_coordinate (float): The y coordinate of the point.
        """
        self._execute_function(
            self.lib.mkernel_curvilinear_delete_node,
            self._meshkernelid,
            c_double(x_coordinate),
            c_double(y_coordinate),
        )

    def _get_num_coordinates(self, location_type):
        """Get the numbers of mesh coordinates of a specific location.

        Args:
            location_type (Mesh2dLocation): The location type.

        Raises:
            Exception: This exception gets raised if an invalid location is used.
        """

        mesh = self.mesh2d_get()

        if location_type == Mesh2dLocation.NODES:
            number_of_coordinates = len(mesh.node_x)
        elif location_type == Mesh2dLocation.FACES:
            number_of_coordinates = len(mesh.face_x)
        elif location_type == Mesh2dLocation.EDGES:
            number_of_coordinates = len(mesh.edge_x)
        else:
            raise ValueError("wrong location_type")

        return number_of_coordinates

    def __map_to_valid_values(
        self, geometry_list: GeometryList, location_type: Mesh2dLocation
    ):
        """Removes invalid values from geometry_list

        Args:
            geometry_list (GeometryList): The input geometry list
            location_type (Mesh2dLocation): The location type

        Raises:
            Exception: This exception gets raised if an invalid location is used.
        """

        mesh = self.mesh2d_get()

        if location_type == location_type.NODES:
            geometry_list.x_coordinates = mesh.node_x
            geometry_list.y_coordinates = mesh.node_y
            geometry_list.values = np.array(
                [geometry_list.values[i] for i in mesh.valid_nodes_map.keys()],
                dtype=np.double,
            )
        elif location_type == location_type.FACES:
            geometry_list.x_coordinates = mesh.face_x
            geometry_list.y_coordinates = mesh.face_y
            geometry_list.values = np.array(
                [geometry_list.values[i] for i in mesh.valid_faces_map.keys()],
                dtype=np.double,
            )
        elif location_type == location_type.EDGES:
            geometry_list.x_coordinates = mesh.edge_x
            geometry_list.y_coordinates = mesh.edge_y
            geometry_list.values = np.array(
                [geometry_list.values[i] for i in mesh.valid_edges_map.keys()],
                dtype=np.double,
            )
        else:
            raise ValueError("wrong location_type")

        return geometry_list
