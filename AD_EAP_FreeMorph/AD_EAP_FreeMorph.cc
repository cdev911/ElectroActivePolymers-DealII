/* ========================
 * Electro Active Polymers
 * ========================
 * Problem description:
 *   Nonlinear electro-elastostastic solver with the consideration of 
 *   electric field in the free space, further the freespace undergo
 *   pseudo morphing due to deformation of the bulk.
 *
 *   Author: Chaitanya Dev
 *           Friedrich-Alexander University Erlangen-Nuremberg
 *
 *  References:
 *  Vu, K., On coupled BEM-FEM simulation of nonlinear electro-elastostatics
 *  Vogel, F., On the modelling and Computation of Electro- and Magneto-Active Polymers
 *  Deal.II 9.3 - Step 72
 */

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/tensor.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/affine_constraints.h>

#include <deal.II/lac/sparse_direct.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/hp/fe_collection.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>

#include <deal.II/integrators/elasticity.h>
#include <deal.II/physics/elasticity/standard_tensors.h>
#include <deal.II/physics/elasticity/kinematics.h>

#include <deal.II/algorithms/general_data_storage.h>
#include <deal.II/differentiation/ad.h>

#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/timer.h>
// This again is C++:
#include <fstream>
#include <iostream>


namespace AD_EAP_FreeMorph
{
  using namespace dealii;

 class Errors {
    /**
     * @brief This class is to compute the relative error, which is used in Newton Raphson scheme. 
     * The class is Initialized with error in the first iteration, subsequently in later iterations given an error value, it can return
     * normalized error wrt the error in first iteration.
     */
  private:
    double error_first_iter = 0.0;
    bool initialized = false;
  public:
  /**
   * @brief Initialize error.
   * 
   * @param error 
   */
    inline void Initialize(double error) {
      if (error == 0.0)
        throw std::runtime_error ("First iteration error cannot be 0.0 ");
      else {
        if (!initialized) {
          error_first_iter = error;
          initialized = true;
        } else
          std::cerr << "Already the error is initialized." << std::endl;
      }
    }

    /**
     * @brief Function to get the Normalized Error. 
     * 
     * @param error 
     * @return double 
     */
    inline double get_normalized_error(double error) {
      if (initialized)
        return error / error_first_iter;
      else {
        std::cerr << "First iteration error not initialized, so cannot Normalize." << std::endl;
        return 1e9;
      }
    }

    /**
     * @brief Reset the error.
     * 
     */
    inline void Reset() {
      error_first_iter = 0.0;
      initialized = false;
    }
  };

  class ParameterReader : public Subscriptor
  {
  public:
    ParameterReader(ParameterHandler &);
    void read_parameters(const std::string &);
    void output_default_parameters(); 
    void declare_parameters();
  private:
    
    ParameterHandler &prm;
  };
 
  ParameterReader::ParameterReader(ParameterHandler &paramhandler)
    : prm(paramhandler)
  {}


   void ParameterReader::output_default_parameters() {
    const std::string out_file = "input.prm";
    std::ofstream parameter_out(out_file);
    prm.print_parameters(parameter_out, ParameterHandler::Text);
  }

   void ParameterReader::declare_parameters()
  {
    prm.enter_subsection("General");
    {
      prm.declare_entry("Problem dimension",
                        "2",
                        Patterns::Integer(2,3),
                        "Space dimension of the problem ");
      prm.declare_entry("Solver",
                        "umfpack",
                        Patterns::Anything(),
                        "Type of solver: currently you can choose"
                        "{cg, umfpack}");
                        
    }
    prm.leave_subsection();

    prm.enter_subsection("Mesh & geometry parameters");
    {
      prm.declare_entry("Geometry type",
                    "hyper_rectangle_hole",
                    Patterns::Anything(),
                    "Type of geometry: currently you can choose"
                    "{hyper_rectangle, hyper_rectangle_hole}");

      prm.declare_entry("Number of refinements",
                        "0",
                        Patterns::Integer(0,10),
                        "Number of global mesh refinement steps "
                        "applied to initial coarse grid");

      prm.declare_entry("Number of subdivision in X",
                        "40",
                        Patterns::Integer(0,1000),
                        "Number of cells in X direction "
                        "applied to initial coarse grid");

      prm.declare_entry("Number of subdivision in Y",
                        "40",
                        Patterns::Integer(0,1000),
                        "Number of cells in Y direction "
                        "applied to initial coarse grid");

      prm.declare_entry("Number of subdivision in Z",
                        "0",
                        Patterns::Integer(0,1000),
                        "Number of cells in Z direction "
                        "applied to initial coarse grid");

      prm.declare_entry("Inner radius",
                        "15e-6",
                        Patterns::Double(),
                        "Center of the plate with a hole in m");

      prm.declare_entry("Outer radius",
                        "30e-6",
                        Patterns::Double(),
                        "Center of the plate with a hole in m");
    }
    prm.leave_subsection();
 
    prm.enter_subsection("Material constants");
    {
      prm.declare_entry("lambda", "0.06e6", Patterns::Double(), "lame 1st parameter in Pa");
 
      prm.declare_entry("mu", "0.05e6", Patterns::Double(), "lame 2nd parameter in Pa");

      prm.declare_entry("c1", "1.7708e-12", Patterns::Double(), "material constant refer Vogel's thesis eq 5.21 in Pa m^2/V^2");

      prm.declare_entry("c2", "1.7708e-11", Patterns::Double(), "material constant refer Vogel's thesis eq 5.21 in Pa m^2/V^2");

      prm.declare_entry("epsilon", "4.425e-11", Patterns::Double(), "relative permitivity in Pa m^2/V^2");
    }
    prm.leave_subsection();
 
    prm.enter_subsection("Boundary conditions");
    {
      prm.declare_entry("voltage", "3e1", Patterns::Double(), "prescribed voltage.");

      prm.declare_entry("Boundary conditions case",
                    "hyper_rectangle_hole",
                    Patterns::Anything(),
                    "Case of boundary condition: currently you can choose"
                    "{hyper_rectangle, hyper_rectangle_edge, plate_with_a_hole}");
    }
    prm.leave_subsection();

    prm.enter_subsection("Newton raphson");
    {
      prm.declare_entry("number load steps", "5", Patterns::Integer(0), "Number of load steps");

      prm.declare_entry("tolerance", "1e-6", Patterns::Double(0), "Residual error tolerance");

      prm.declare_entry("max newton iter", "30", Patterns::Integer(0), "Max number of newton iterations");
    }
    prm.leave_subsection();


    prm.enter_subsection("Output filename");
    {
      prm.declare_entry("Output filename",
                    "solution",
                    Patterns::Anything(),
                    "Name of the output file (without extension)");
    }
    prm.leave_subsection();
 
  }
 
 
  void ParameterReader::read_parameters(const std::string &parameter_file)
  {
    declare_parameters();
 
    prm.parse_input(parameter_file);
  }

  enum
  {
    solid_id,
    void_id
  };

  template <int dim>
  inline bool 
  cell_is_bulk(const typename DoFHandler<dim>::cell_iterator &cell)
  {
    return (cell->material_id() == solid_id);
  }

  template <int dim>
  inline bool 
  cell_is_free(const typename DoFHandler<dim>::cell_iterator &cell)
  {
    return (cell->material_id() == void_id);
  }


  template <int dim>
  void hyper_rectangle_hole(ParameterHandler &prm,
                        Triangulation<dim> &triangulation,
                        Vector<double> &tria_boundary_ids,
                        Vector<double> &tria_material_ids)
  {
    prm.enter_subsection("Mesh & geometry parameters");
    const double inner_radius = prm.get_double("Inner radius");
    const double outer_radius = prm.get_double("Outer radius");
    const unsigned int refinement = prm.get_integer("Number of refinements");
    const unsigned int  subdiv_x = prm.get_integer("Number of subdivision in X");
    const unsigned int  subdiv_y = prm.get_integer("Number of subdivision in Y");
    const unsigned int  subdiv_z = prm.get_integer("Number of subdivision in X");
    prm.leave_subsection();

    const double free = outer_radius*2.;

    if(dim == 2)
    {
      std::vector<unsigned int> subdivision = {subdiv_x, subdiv_y};
      Point<dim> p1(-free,-free), p2(free,free);
      GridGenerator::subdivided_hyper_rectangle(triangulation, subdivision, p1, p2, true);
      triangulation.refine_global(refinement); 

      tria_boundary_ids.reinit(triangulation.n_active_cells());
      tria_material_ids.reinit(triangulation.n_active_cells());
      Vector<double> dom_boundary_ids(triangulation.n_active_cells() * GeometryInfo<dim>::faces_per_cell);
      unsigned int counter = 0, cell_counter = 0;
      for (auto cell: triangulation.active_cell_iterators()) {
        Point<dim> center = cell->center();
        if(std::abs(center[0]) < inner_radius && std::abs(center[1]) < inner_radius){
          cell->set_material_id(void_id);
          tria_material_ids[cell_counter] = void_id;
        } 
        else if(std::abs(center[0]) < outer_radius && std::abs(center[1]) < outer_radius  ){
          cell->set_material_id(solid_id);
          tria_material_ids[cell_counter] = solid_id;
        }
        else{
          cell->set_material_id(void_id);
          tria_material_ids[cell_counter] = void_id;
        }
        if (cell->at_boundary())
          for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f) {
            if (cell->face(f)->at_boundary()) {
              int boundary_id = cell->face(f)->boundary_id();
              if (dom_boundary_ids[counter] == 0) {
                dom_boundary_ids[counter] = boundary_id;
                tria_boundary_ids[cell_counter] = boundary_id;
              }
            }
            ++counter;
          }
        ++cell_counter;
      }
    } // if dim == 2
    else if(dim == 3)
    {
      std::vector<unsigned int> subdivision = {subdiv_x, subdiv_y, subdiv_z};
      Point<dim> p1(-free,-free,-free), p2(free,free,free);
      GridGenerator::subdivided_hyper_rectangle(triangulation, subdivision, p1, p2, true);
      triangulation.refine_global(refinement); 

      tria_boundary_ids.reinit(triangulation.n_active_cells());
      tria_material_ids.reinit(triangulation.n_active_cells());
      Vector<double> dom_boundary_ids(triangulation.n_active_cells() * GeometryInfo<dim>::faces_per_cell);
      unsigned int counter = 0, cell_counter = 0;
      for (auto cell: triangulation.active_cell_iterators()) {
        Point<dim> center = cell->center();
        if(std::abs(center[0]) < inner_radius && std::abs(center[1]) < inner_radius ){
          cell->set_material_id(void_id);
          tria_material_ids[cell_counter] = void_id;
        } 
        else if(std::abs(center[0]) < outer_radius && std::abs(center[1]) < outer_radius && std::abs(center[2]) < outer_radius  ){
          cell->set_material_id(solid_id);
          tria_material_ids[cell_counter] = solid_id;
        }
        else{
          cell->set_material_id(void_id);
          tria_material_ids[cell_counter] = void_id;
        }
        if (cell->at_boundary())
          for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f) {
            if (cell->face(f)->at_boundary()) {
              int boundary_id = cell->face(f)->boundary_id();
              if (dom_boundary_ids[counter] == 0) {
                dom_boundary_ids[counter] = boundary_id;
                tria_boundary_ids[cell_counter] = boundary_id;
              }
            }
            ++counter;
          }
        ++cell_counter;
      }
    }
  }

  template <int dim>
  void hyper_rectangle(ParameterHandler &prm,
                        Triangulation<dim> &triangulation,
                        Vector<double> &tria_boundary_ids,
                        Vector<double> &tria_material_ids)
  {
    prm.enter_subsection("Mesh & geometry parameters");
    const unsigned int refinement = prm.get_integer("Number of refinements");
    prm.leave_subsection();
    const double bulk = 30e-6;
    const double free = bulk*4.;
    if(dim == 2)
    {
      Point<dim> p1(-free,-free), p2(free,free);
      GridGenerator::hyper_rectangle(triangulation, p1, p2, true);
      triangulation.refine_global(refinement); 

      tria_boundary_ids.reinit(triangulation.n_active_cells());
      tria_material_ids.reinit(triangulation.n_active_cells());
      Vector<double> dom_boundary_ids(triangulation.n_active_cells() * GeometryInfo<dim>::faces_per_cell);
      unsigned int counter = 0, cell_counter = 0;
      for (auto cell: triangulation.active_cell_iterators()) {
        Point<dim> center = cell->center();
        if(std::abs(center[0]) < bulk && std::abs(center[1]) < bulk  ){
          cell->set_material_id(solid_id);
          tria_material_ids[cell_counter] = solid_id;
        }
        else{
          cell->set_material_id(void_id);
          tria_material_ids[cell_counter] = void_id;
        }
        if (cell->at_boundary())
          for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f) {
            if (cell->face(f)->at_boundary()) {
              int boundary_id = cell->face(f)->boundary_id();
              if (dom_boundary_ids[counter] == 0) {
                dom_boundary_ids[counter] = boundary_id;
                tria_boundary_ids[cell_counter] = boundary_id;
              }
            }
            ++counter;
          }
        ++cell_counter;
      }
    } 
    else if(dim == 3)
    {
      Point<dim> p1(-free,-free,-free), p2(free,free,free);
      GridGenerator::hyper_rectangle(triangulation, p1, p2, true);
      triangulation.refine_global(refinement); 

      tria_boundary_ids.reinit(triangulation.n_active_cells());
      tria_material_ids.reinit(triangulation.n_active_cells());
      Vector<double> dom_boundary_ids(triangulation.n_active_cells() * GeometryInfo<dim>::faces_per_cell);
      unsigned int counter = 0, cell_counter = 0;
      for (auto cell: triangulation.active_cell_iterators()) {
        Point<dim> center = cell->center();
        if(std::abs(center[0]) < bulk && std::abs(center[1]) < bulk && std::abs(center[2]) < bulk  ){
          cell->set_material_id(solid_id);
          tria_material_ids[cell_counter] = solid_id;
        }
        else{
          cell->set_material_id(void_id);
          tria_material_ids[cell_counter] = void_id;
        }
        if (cell->at_boundary())
          for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f) {
            if (cell->face(f)->at_boundary()) {
              int boundary_id = cell->face(f)->boundary_id();
              if (dom_boundary_ids[counter] == 0) {
                dom_boundary_ids[counter] = boundary_id;
                tria_boundary_ids[cell_counter] = boundary_id;
              }
            }
            ++counter;
          }
        ++cell_counter;
      }
    }
    
  }
 
  template <int dim>
  void plate_with_a_hole(ParameterHandler &prm,
                        Triangulation<dim> &triangulation,
                        Vector<double> &tria_boundary_ids,
                        Vector<double> &tria_material_ids)
  {
    if(dim == 2){
      prm.enter_subsection("Mesh & geometry parameters");
      const double inner_radius = prm.get_double("Inner radius");
      const double outer_radius = prm.get_double("Outer radius");
      const unsigned int refinement = prm.get_integer("Number of refinements");
      prm.leave_subsection();
      double pad = inner_radius*5;

      Point<dim> center; center[0] =0; center[1] =0;
      GridGenerator::plate_with_a_hole(triangulation,
        inner_radius, outer_radius, pad, pad, pad, pad, center, 10, 20, 1, 2, true);
      triangulation.refine_global(refinement); 

      tria_boundary_ids.reinit(triangulation.n_active_cells());
      tria_material_ids.reinit(triangulation.n_active_cells());
      Vector<double> dom_boundary_ids(triangulation.n_active_cells() * GeometryInfo<dim>::faces_per_cell);
      unsigned int counter = 0, cell_counter = 0;
      for (auto cell: triangulation.active_cell_iterators()) {
        Point<dim> center = cell->center();
        if(std::abs(center[0]) < outer_radius && std::abs(center[1]) < outer_radius  ){
          cell->set_material_id(solid_id);
          tria_material_ids[cell_counter] = solid_id;
        }
        else{
          cell->set_material_id(void_id);
          tria_material_ids[cell_counter] = void_id;
        }
        if (cell->at_boundary())
          for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f) {
            if (cell->face(f)->at_boundary()) {
              int boundary_id = cell->face(f)->boundary_id();
              if (dom_boundary_ids[counter] == 0) {
                dom_boundary_ids[counter] = boundary_id;
                tria_boundary_ids[cell_counter] = boundary_id;
              }
            }
            ++counter;
          }
        ++cell_counter;
      }
    } 
    else {
      std::string error_msg = std::string(__FILE__) + ":" + std::to_string(__LINE__) + " does not work for 3D!";
      throw std::runtime_error(error_msg);
    }
  }

  template <int dim>
  void setup_dirichlet_dof_info_hyper_rectangle(ParameterHandler &prm,
                                                DoFHandler<dim> &dof_handler,
                                                std::map<unsigned int, double> &map_dir_dof_index_to_val)
  {
    map_dir_dof_index_to_val.clear();

    const double bulk = 30e-6;

    prm.enter_subsection("Boundary conditions");
    const double voltage = prm.get_double("voltage");
    prm.leave_subsection();

    for (auto cell: dof_handler.active_cell_iterators()) {

      if(cell_is_free<dim>(cell))
        continue;

      for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell; ++v) {
        
        Point<dim> vertex = cell->vertex(v);
        if(std::abs(std::abs(vertex[1]) - bulk) < 1e-10){
          unsigned int dof_index = cell->vertex_dof_index(v, dim /* phi-dof */, cell->active_fe_index());
          if(vertex[1] < 0.0){
            map_dir_dof_index_to_val[dof_index] = -1.0*voltage; // dir_value
          }
          else{
            map_dir_dof_index_to_val[dof_index] = voltage; // dir_value
          }
        }

        if(dim == 2){
          if(std::abs(vertex[0] + bulk) < 1e-10 && std::abs(vertex[1]) < bulk){
            unsigned int dof_index = cell->vertex_dof_index(v, 0, cell->active_fe_index());
            map_dir_dof_index_to_val[dof_index] = 0.0; // dir_value
          }
          if(std::abs(vertex[1] + bulk) < 1e-10 && std::abs(vertex[0]) < bulk){
            unsigned int dof_index = cell->vertex_dof_index(v, 1, cell->active_fe_index());
            map_dir_dof_index_to_val[dof_index] = 0.0; // dir_value
          }
        } else if(dim == 3){
          if(std::abs(vertex[0] + bulk) < 1e-10 && std::abs(vertex[1] + bulk) < 1e-10 && std::abs(vertex[2]) < bulk){
            unsigned int dof_index_0 = cell->vertex_dof_index(v, 0, cell->active_fe_index());
            map_dir_dof_index_to_val[dof_index_0] = 0.0; // dir_value
            unsigned int dof_index_1 = cell->vertex_dof_index(v, 1, cell->active_fe_index());
            map_dir_dof_index_to_val[dof_index_1] = 0.0; // dir_value
          }
          if(std::abs(vertex[1] + bulk) < 1e-10 && std::abs(vertex[2] + bulk) < 1e-10 && std::abs(vertex[0]) < bulk){
            unsigned int dof_index_1 = cell->vertex_dof_index(v, 1, cell->active_fe_index());
            map_dir_dof_index_to_val[dof_index_1] = 0.0; // dir_value
            unsigned int dof_index_2 = cell->vertex_dof_index(v, 2, cell->active_fe_index());
            map_dir_dof_index_to_val[dof_index_2] = 0.0; // dir_value
          }
          if(std::abs(vertex[2] + bulk) < 1e-10 && std::abs(vertex[0] + bulk) < 1e-10 && std::abs(vertex[1]) < bulk){
            unsigned int dof_index_2 = cell->vertex_dof_index(v, 2, cell->active_fe_index());
            map_dir_dof_index_to_val[dof_index_2] = 0.0; // dir_value
            unsigned int dof_index_0 = cell->vertex_dof_index(v, 1, cell->active_fe_index());
            map_dir_dof_index_to_val[dof_index_0] = 0.0; // dir_value
          }
        }

      }
    }

  }

  
  template <int dim>
  void setup_dirichlet_dof_info_hyper_rectangle_hole(ParameterHandler &prm,
                                                DoFHandler<dim> &dof_handler,
                                                std::map<unsigned int, double> &map_dir_dof_index_to_val)
  {
    prm.enter_subsection("Boundary conditions");
    const double voltage = prm.get_double("voltage");
    prm.leave_subsection();

    prm.enter_subsection("Mesh & geometry parameters");
    const double outer_radius = prm.get_double("Outer radius");
    prm.leave_subsection();

    for (auto cell: dof_handler.active_cell_iterators()) {

      if(cell_is_free<dim>(cell))
        continue;

      for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell; ++v) {
        
        Point<dim> vertex = cell->vertex(v);
        if(std::abs(std::abs(vertex[1]) - outer_radius) < 1e-10){
          unsigned int dof_index = cell->vertex_dof_index(v, dim /* phi-dof */, cell->active_fe_index());
          if(vertex[1] < 0.0){
            map_dir_dof_index_to_val[dof_index] = -1.0*voltage; // dir_value
          }
          else{
            map_dir_dof_index_to_val[dof_index] = voltage; // dir_value
          }
        }

        if(std::abs(std::abs(vertex[0]) - outer_radius) < outer_radius*0.15 && 
           std::abs(std::abs(vertex[1]) - outer_radius) < 1e-10)
        {
          if(dim == 2){
            for(unsigned int d = 0; d < dim; ++d){
              unsigned int dof_index = cell->vertex_dof_index(v, d, cell->active_fe_index());
              map_dir_dof_index_to_val[dof_index] = 0.0; // dir_value
            }
          }
          else if(dim == 3 && std::abs(vertex[2]) < outer_radius){
            for(unsigned int d = 0; d < dim; ++d){
              unsigned int dof_index = cell->vertex_dof_index(v, d, cell->active_fe_index());
              map_dir_dof_index_to_val[dof_index] = 0.0; // dir_value
            }
          }
        }
      }
    }
  }

  template <int dim>
  void setup_dirichlet_dof_info_hyper_rectangle_edge(ParameterHandler &prm,
                                                DoFHandler<dim> &dof_handler,
                                                std::map<unsigned int, double> &map_dir_dof_index_to_val)
  {
    const double bulk = 30e-6;
    prm.enter_subsection("Boundary conditions");
    const double voltage = prm.get_double("voltage");
    prm.leave_subsection();

    for (auto cell: dof_handler.active_cell_iterators()) {

      if(cell_is_free<dim>(cell))
        continue;

      for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell; ++v) {
        
        Point<dim> vertex = cell->vertex(v);
        if(std::abs(std::abs(vertex[1]) - bulk) < 1e-10){
          unsigned int dof_index = cell->vertex_dof_index(v, dim /* phi-dof */, cell->active_fe_index());
          if(vertex[1] < 0.0){
            map_dir_dof_index_to_val[dof_index] = -1.0*voltage; // dir_value
          }
          else{
            map_dir_dof_index_to_val[dof_index] = voltage; // dir_value
          }
        }

        if(std::abs(std::abs(vertex[0]) - bulk) < bulk*0.15 && 
           std::abs(std::abs(vertex[1]) - bulk) < 1e-10)
        {
          if(dim == 2){
            for(unsigned int d = 0; d < dim; ++d){
              unsigned int dof_index = cell->vertex_dof_index(v, d, cell->active_fe_index());
              map_dir_dof_index_to_val[dof_index] = 0.0; // dir_value
            }
          }
          else if(dim == 3 && std::abs(vertex[2]) < bulk){
            for(unsigned int d = 0; d < dim; ++d){
              unsigned int dof_index = cell->vertex_dof_index(v, d, cell->active_fe_index());
              map_dir_dof_index_to_val[dof_index] = 0.0; // dir_value
            }
          }

        }
      }
    }

  }

  template <int dim>
  void setup_dirichlet_dof_info_plate_with_hole(ParameterHandler &prm,
                                                DoFHandler<dim> &dof_handler,
                                                std::map<unsigned int, double> &map_dir_dof_index_to_val)
  {

    prm.enter_subsection("Boundary conditions");
    const double voltage = prm.get_double("voltage");
    prm.leave_subsection();

    prm.enter_subsection("Mesh & geometry parameters");
    double inner_radius = prm.get_double("Inner radius");
    double outer_radius = prm.get_double("Outer radius");
    prm.leave_subsection();

    Point<dim> pt_temp; 
    std::vector<Point<dim>> hor_points_const_y, ver_points_const_x;
    pt_temp[1] = 0.0;
    pt_temp[0] = -1.0*inner_radius;  
    hor_points_const_y.push_back(pt_temp);
    pt_temp[0] = inner_radius; 
    hor_points_const_y.push_back(pt_temp);
    
    pt_temp[0] = 0.0;
    pt_temp[1] = -1.0*inner_radius;  
    ver_points_const_x.push_back(pt_temp);
    pt_temp[1] = inner_radius; 
    ver_points_const_x.push_back(pt_temp);

    for (auto cell: dof_handler.active_cell_iterators()) {

      if(cell_is_free<dim>(cell))
        continue;

      for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell; ++v) {
        
        Point<dim> vertex = cell->vertex(v);
        if(std::abs(std::abs(vertex[1]) - outer_radius) < 1e-10){
          unsigned int dof_index = cell->vertex_dof_index(v, dim /* phi-dof */, cell->active_fe_index());
          if(vertex[1] < 0.0){
            map_dir_dof_index_to_val[dof_index] = -1.0*voltage; // dir_value
          }
          else{
            map_dir_dof_index_to_val[dof_index] = voltage; // dir_value
          }
          continue;
        }

        if(std::find_if(
          hor_points_const_y.begin(), hor_points_const_y.end(),
          [&vertex](const Point<dim>& pt) { return pt.distance(vertex) < 1e-12;}) 
          != hor_points_const_y.end()){
            unsigned int dof_index = cell->vertex_dof_index(v, 1 /* y-dof */, cell->active_fe_index());
            map_dir_dof_index_to_val[dof_index] = 0.0; // dir_value

            continue;
          }
        if(std::find_if(
          ver_points_const_x.begin(), ver_points_const_x.end(),
          [&vertex](const Point<dim>& pt) { return pt.distance(vertex) < 1e-12;}) 
          != ver_points_const_x.end()){
            unsigned int dof_index = cell->vertex_dof_index(v, 0 /* x-dof */, cell->active_fe_index());
            map_dir_dof_index_to_val[dof_index] = 0.0; // dir_value

            continue;
          }

      }
    }
  }

  template <int dim>
  struct cell_interface_data{
    std::vector<Point<dim>> interface_support_point;
    std::vector<std::vector<unsigned int>> vec_interface_dof_indices;
  };

  template <int dim, Differentiation::AD::NumberTypes ADTypeCode>
  class EAPFreeMorphProblem
  {
  public:
    EAPFreeMorphProblem(ParameterHandler &);
    void run();

  private:

    void make_grid();
    void set_fe_indices();
    void setup_system();
    void setup_constraint();
    void setup_dirichlet_dof_info();
    void assemble_system();
    void solve(Vector<double> &newton_update);
    void solve_load_step_NR();
    void output_results(const unsigned int cycle) const;

    Vector<double> get_total_solution(const Vector<double> &solution_delta) const;
    double get_error_residual();

    void setup_constraint_mesh();
    void assemble_system_mesh();
    void solve_mesh();

    ParameterHandler &prm;

    Triangulation<dim> triangulation;
    DoFHandler<dim>    dof_handler;
    Vector<double> tria_boundary_ids;
    Vector<double> tria_material_ids;

    FESystem<dim> fe_bulk;
    FESystem<dim> fe_free;
    hp::FECollection<dim> fe_collection;

    double lambda, mu, c_1, c_2, epsilon;

    AffineConstraints<double> constraints;
    std::map<unsigned int, double> map_dir_dof_index_to_val;

    SparsityPattern      sparsity_pattern;
    SparseMatrix<double> system_matrix;

    Vector<double> solution;
    Vector<double> solution_delta;
    Vector<double> newton_update;
    Vector<double> residual;

    unsigned int newton_iteration = 0;
    unsigned int max_nr_steps = 20;
    double curr_load = 1.0;
    double accum_load = 0.0;
    double init_load = 1.0;
    int load_step;
    unsigned int number_load_steps = 5;
    double nr_tolerance = 1.0e-8;
    int terminate_loadstep = 0;

    Errors error_NR;


    DoFHandler<dim>    dof_handler_mesh;

    FESystem<dim> fe_mesh;
    FESystem<dim> fe_void_mesh;
    hp::FECollection<dim> fe_collection_mesh;

    AffineConstraints<double> constraints_mesh;

    SparsityPattern      sparsity_pattern_mesh;
    SparseMatrix<double> system_matrix_mesh;

    Vector<double> solution_mesh, system_rhs_mesh;

    std::map<unsigned int, unsigned int> map_mesh_dofindex_to_elastic_dofindex;

    std::map<CellId, cell_interface_data<dim>> mesh_exchange_data;

    mutable TimerOutput compute_timer;
  };



  template <int dim, Differentiation::AD::NumberTypes ADTypeCode>
  EAPFreeMorphProblem<dim, ADTypeCode>::EAPFreeMorphProblem(ParameterHandler &prm_)
    : prm(prm_)
    , dof_handler(triangulation)
    , fe_bulk(FE_Q<dim>(1), dim,
         FE_Q<dim>(1), 1)
    , fe_free(FE_Nothing<dim>(), dim,
              FE_Q<dim>(1), 1)
    , dof_handler_mesh(triangulation)
    , fe_mesh(FE_Q<dim>(1), dim)
    , fe_void_mesh(FE_Nothing<dim>(), dim)
    , compute_timer(std::cout, TimerOutput::summary, TimerOutput::wall_times)

  {
    fe_collection.push_back(fe_bulk);
    fe_collection.push_back(fe_free);

    fe_collection_mesh.push_back(fe_mesh);
    fe_collection_mesh.push_back(fe_void_mesh);

    prm.enter_subsection("Material constants");
      lambda = prm.get_double("lambda");
      mu = prm.get_double("mu");
      c_1 = prm.get_double("c1");
      c_2 = prm.get_double("c2");
      epsilon = prm.get_double("epsilon");
    prm.leave_subsection();

    prm.enter_subsection("Newton raphson");
    
      max_nr_steps = prm.get_integer("max newton iter"); 
      nr_tolerance = prm.get_double("tolerance"); 
      number_load_steps = prm.get_integer("number load steps"); prm.declare_entry("max newton iter", "20", Patterns::Integer(0), "Max number of newton iterations");
    prm.leave_subsection();
  }


  template <int dim, Differentiation::AD::NumberTypes ADTypeCode>
  void EAPFreeMorphProblem<dim, ADTypeCode>::set_fe_indices()
  {
    // For standard problem
    for (const auto &cell : dof_handler.active_cell_iterators())
    {
      if (cell_is_bulk<dim>(cell))
        cell->set_active_fe_index(solid_id);
      else if (cell_is_free<dim>(cell))
        cell->set_active_fe_index(void_id);
      else
        Assert(false, ExcNotImplemented());
    }

    // For the mesh problem
    for (const auto &cell : dof_handler_mesh.active_cell_iterators())
    {
      if (cell_is_bulk<dim>(cell))
        cell->set_active_fe_index(void_id);
      else if (cell_is_free<dim>(cell))
        cell->set_active_fe_index(solid_id);
      else
        Assert(false, ExcNotImplemented());
    }
  }

  template <int dim, Differentiation::AD::NumberTypes ADTypeCode>
  void EAPFreeMorphProblem<dim, ADTypeCode>::setup_system()
  {
    TimerOutput::Scope t(compute_timer, "setup_system");
    set_fe_indices();

    // For the standard problem
    dof_handler.distribute_dofs(fe_collection);

    solution.reinit(dof_handler.n_dofs());
    solution_delta.reinit(dof_handler.n_dofs());
    newton_update.reinit(dof_handler.n_dofs());
    residual.reinit(dof_handler.n_dofs());

    DynamicSparsityPattern dsp(dof_handler.n_dofs(), dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler,
                                    dsp,
                                    constraints,
                                    /*keep_constrained_dofs = */ false);
    sparsity_pattern.copy_from(dsp);

    system_matrix.reinit(sparsity_pattern);

    setup_dirichlet_dof_info();

    for (const auto &cell : dof_handler.active_cell_iterators())
    {
      if (cell_is_bulk<dim>(cell))
      {
        for (const auto face_no : cell->face_indices())
        {
          if(cell->at_boundary(face_no) == true)
            continue;
          if ((cell->neighbor(face_no)->has_children() == false) &&
              (cell_is_free<dim>(cell->neighbor(face_no))))
          {
            cell_interface_data<dim> temp;
            for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_face; ++v) {
              unsigned int cell_vertex_idx = GeometryInfo< dim >::face_to_cell_vertices(face_no, v);	
              Point<dim> vertex = cell->vertex(cell_vertex_idx);
              temp.interface_support_point.push_back(vertex);
              std::vector<unsigned int> point_dof_indices;
              for (unsigned int d = 0; d < dim; ++d) {
                unsigned int dof_index = cell->vertex_dof_index(cell_vertex_idx, d, cell->active_fe_index());
                point_dof_indices.push_back(dof_index);
              }
              temp.vec_interface_dof_indices.push_back(point_dof_indices);
            } // end of loop over vertices_per_face
            mesh_exchange_data[cell->neighbor(face_no)->id()] = temp;
          }
          else if (cell->neighbor(face_no)->has_children() == true)
          {
            for (unsigned int sf = 0;
                sf < cell->face(face_no)->n_children();
                ++sf)
            {
              if (cell_is_free<dim>(
                    cell->neighbor_child_on_subface(face_no, sf)))
              {

                cell_interface_data<dim> temp;
                for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_face; ++v) {
                  unsigned int cell_vertex_idx = GeometryInfo< dim >::face_to_cell_vertices(face_no, v);
                  Point<dim> vertex = cell->vertex(cell_vertex_idx);
                  temp.interface_support_point.push_back(vertex);
                  std::vector<unsigned int> point_dof_indices;
                  for (unsigned int d = 0; d < dim; ++d) {
                    unsigned int dof_index = cell->vertex_dof_index(cell_vertex_idx, d, cell->active_fe_index());
                    point_dof_indices.push_back(dof_index);
                  }
                  temp.vec_interface_dof_indices.push_back(point_dof_indices);
                } // end of loop over vertices_per_face
                mesh_exchange_data[cell->neighbor_child_on_subface(face_no, sf)->id()] = temp;
                break;
              }
            }
          }
        } // end of loop over cell->face_indices()
      } // if cell is solid
    } // end of loop over cells

    // For the mesh problem
    dof_handler_mesh.distribute_dofs(fe_collection_mesh);

    solution_mesh.reinit(dof_handler_mesh.n_dofs());
    system_rhs_mesh.reinit(dof_handler_mesh.n_dofs());

    DynamicSparsityPattern dsp_mesh(dof_handler_mesh.n_dofs(), dof_handler_mesh.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler_mesh,
                                    dsp_mesh,
                                    constraints_mesh,
                                    /*keep_constrained_dofs = */ false);
    sparsity_pattern_mesh.copy_from(dsp_mesh);

    system_matrix_mesh.reinit(sparsity_pattern_mesh);

    for (const auto &cell : dof_handler_mesh.active_cell_iterators())
    {
      if (cell_is_free<dim>(cell))
      {
        for (const auto face_no : cell->face_indices())
        {
          if(cell->at_boundary(face_no) == true)
            continue;
          if ((cell->neighbor(face_no)->has_children() == false) &&
              (cell_is_bulk<dim>(cell->neighbor(face_no))))
          {
            cell_interface_data<dim> temp = mesh_exchange_data[cell->id()];
            for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_face; ++v) {
              unsigned int cell_vertex_idx = GeometryInfo< dim >::face_to_cell_vertices(face_no, v);	
              Point<dim> vertex = cell->vertex(cell_vertex_idx);
              // Find the index corresponding to the point
              auto pos = std::find_if(temp.interface_support_point.cbegin(), 
                temp.interface_support_point.cend(), 
                [&vertex](Point<dim> pt){return (pt.distance_square(vertex) < 1e-12);});
              unsigned int my_data_idx = std::distance(temp.interface_support_point.cbegin(), pos);
              if(pos == temp.interface_support_point.cend())
                continue;
              // Get solution dof index
              std::vector<unsigned int> sol_dof_indices = temp.vec_interface_dof_indices[my_data_idx];
              for (unsigned int d = 0; d < dim; ++d) {
                unsigned int dof_index = cell->vertex_dof_index(cell_vertex_idx, d, cell->active_fe_index());
                map_mesh_dofindex_to_elastic_dofindex[dof_index] = sol_dof_indices[d];
              }
            } // end of loop over vertices_per_face
          }
          else if (cell->neighbor(face_no)->has_children() == true)
          {
            for (unsigned int sf = 0;
                sf < cell->face(face_no)->n_children();
                ++sf)
              if (cell_is_bulk<dim>(
                    cell->neighbor_child_on_subface(face_no, sf)))
                {
                  cell_interface_data<dim> temp = mesh_exchange_data[cell->id()];
                  for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_face; ++v) {
                    unsigned int cell_vertex_idx = GeometryInfo< dim >::face_to_cell_vertices(face_no, v);	
                    Point<dim> vertex = cell->vertex(cell_vertex_idx);
                    // Find the index corresponding to the point
                    auto pos = std::find_if(temp.interface_support_point.cbegin(), 
                      temp.interface_support_point.cend(), 
                      [&vertex](Point<dim> pt){return (pt.distance_square(vertex) < 1e-10);});
                    unsigned int my_data_idx = std::distance(temp.interface_support_point.cbegin(), pos);
                    // Get solution dof index
                    std::vector<unsigned int> sol_dof_indices = temp.vec_interface_dof_indices[my_data_idx];
                    for (unsigned int d = 0; d < dim; ++d) {
                      unsigned int dof_index = cell->vertex_dof_index(cell_vertex_idx, d, cell->active_fe_index());
                      map_mesh_dofindex_to_elastic_dofindex[dof_index] = sol_dof_indices[d];
                    }
                  } // end of loop over vertices_per_face
                  break;
                }
          }
        } // end of loop over cell->face_indices()
      } // if cell is solid
    } // end of loop over cells
  }

  template <int dim, Differentiation::AD::NumberTypes ADTypeCode>
  void EAPFreeMorphProblem<dim, ADTypeCode>::setup_dirichlet_dof_info()
  {

    prm.enter_subsection("Boundary conditions");
    const std::string boundary_case = prm.get("Boundary conditions case");
    prm.leave_subsection();

    if(boundary_case == "plate_with_a_hole")
      setup_dirichlet_dof_info_plate_with_hole<dim>(prm, dof_handler, map_dir_dof_index_to_val);
    else if(boundary_case == "hyper_rectangle")
      setup_dirichlet_dof_info_hyper_rectangle<dim>(prm, dof_handler, map_dir_dof_index_to_val);
    else if(boundary_case == "hyper_rectangle_edge")
      setup_dirichlet_dof_info_hyper_rectangle_edge<dim>(prm, dof_handler, map_dir_dof_index_to_val);
    else if(boundary_case == "hyper_rectangle_hole")
      setup_dirichlet_dof_info_hyper_rectangle_hole<dim>(prm, dof_handler, map_dir_dof_index_to_val);
    else
    {
      std::string error_msg = std::string(__FILE__) + ":" + std::to_string(__LINE__) + " wrong boundary case!";
      throw std::runtime_error(error_msg);
    }

  }

  template <int dim, Differentiation::AD::NumberTypes ADTypeCode>
  void EAPFreeMorphProblem<dim, ADTypeCode>::setup_constraint()
  {
    TimerOutput::Scope t(compute_timer, "setup_constraint");

    constraints.clear();
    DoFTools::make_hanging_node_constraints(dof_handler, constraints);

    const bool apply_dirichlet_bc = (newton_iteration == 0);

    if (apply_dirichlet_bc) {

      for(auto &[dof_index, dir_value] : map_dir_dof_index_to_val){
        double step_fraction = curr_load;
        double current_load = dir_value * step_fraction;
        constraints.add_line(dof_index);
        constraints.set_inhomogeneity(dof_index, current_load);
      }
      
    } else {
      for(auto &[dof_index, dir_value]: map_dir_dof_index_to_val){
        constraints.add_line(dof_index);
      }
    }
                                             
    constraints.close();
  }

  template <int dim, Differentiation::AD::NumberTypes ADTypeCode>
  void EAPFreeMorphProblem<dim, ADTypeCode>::setup_constraint_mesh()
  {
    TimerOutput::Scope t(compute_timer, "setup_constraint_mesh");

    constraints_mesh.clear();

    DoFTools::make_zero_boundary_constraints(dof_handler_mesh,constraints_mesh);
    DoFTools::make_hanging_node_constraints(dof_handler_mesh, constraints_mesh);

    for(auto &[mesh_dof_index, elastic_dof_index] : map_mesh_dofindex_to_elastic_dofindex){
      constraints_mesh.add_line(mesh_dof_index);
      constraints_mesh.set_inhomogeneity(mesh_dof_index, solution[elastic_dof_index]);
    }

    constraints_mesh.close();

  }

  template <int dim, Differentiation::AD::NumberTypes ADTypeCode>
  void EAPFreeMorphProblem<dim, ADTypeCode>::assemble_system_mesh()
  {
    TimerOutput::Scope t(compute_timer, "assemble_system_mesh");

    system_matrix_mesh = 0.0;
    system_rhs_mesh = 0.0;
    solution_mesh = 0.0;

    QGauss<dim> quadrature_formula(fe_mesh.degree + 1);

    FEValues<dim> fe_values(fe_mesh,
                            quadrature_formula,
                            update_values | update_gradients |
                            update_quadrature_points | update_JxW_values);

    const unsigned int dofs_per_cell = fe_mesh.dofs_per_cell;
    const unsigned int n_q_points    = quadrature_formula.size();

    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double>     cell_rhs(dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
    const FEValuesExtractors::Vector u_fe(0);

    for (const auto &cell : dof_handler_mesh.active_cell_iterators())
    {

      if(cell_is_bulk<dim>(cell))
        continue;
        
      cell_matrix = 0;
      cell_rhs    = 0;

      fe_values.reinit(cell);

      for (unsigned int q = 0; q < n_q_points; ++q) {
        for (unsigned int i = 0; i < dofs_per_cell; ++i) {
          Tensor<2, dim> Grad_N_i = fe_values[u_fe].gradient(i, q);
          for (unsigned int j = 0; j < dofs_per_cell; ++j) {
            Tensor<2, dim> Grad_N_j = fe_values[u_fe].gradient(j, q);
            cell_matrix(i, j) += scalar_product(Grad_N_i,Grad_N_j)* fe_values.JxW(q); // contract3(Grad_N_i, elasticity_tensor, Grad_N_j) * fe_values.JxW(q);
          } // end of j loop
        } // end of i loop
      } // end of loop over quadrature points

      cell->get_dof_indices(local_dof_indices);
      constraints_mesh.distribute_local_to_global(
        cell_matrix, cell_rhs, local_dof_indices, system_matrix_mesh, system_rhs_mesh);
    }
  }


  template <int dim, Differentiation::AD::NumberTypes ADTypeCode>
  void EAPFreeMorphProblem<dim, ADTypeCode>::solve_mesh()
  {
    TimerOutput::Scope t(compute_timer, "solve_mesh");

    SolverControl            solver_control(1000, 1e-12);
    SolverCG<Vector<double>> cg(solver_control);

    PreconditionSSOR<SparseMatrix<double>> preconditioner;
    preconditioner.initialize(system_matrix_mesh, 1.2);

    cg.solve(system_matrix_mesh, solution_mesh, system_rhs_mesh, preconditioner);

    constraints_mesh.distribute(solution_mesh);
  }


  template <int dim, Differentiation::AD::NumberTypes ADTypeCode>
  void EAPFreeMorphProblem<dim, ADTypeCode>::assemble_system()
  {
    TimerOutput::Scope t(compute_timer, "assemble_system");

    using ADHelper = Differentiation::AD::EnergyFunctional<
    Differentiation::AD::NumberTypes::sacado_dfad_dfad,
    double>;
    using ADNumberType = typename ADHelper::ad_type;

    Vector<double> current_solution = get_total_solution(solution_delta);

    QGauss<dim> quadrature_formula(fe_bulk.degree + 1);

    FEValues<dim> fe_values_bulk(fe_bulk,
                            quadrature_formula,
                            update_values | update_gradients |
                            update_quadrature_points | update_JxW_values);

    FEValues<dim> fe_values_free(fe_free,
                            quadrature_formula,
                            update_values | update_gradients |
                            update_quadrature_points | update_JxW_values);

    FEValues<dim> fe_values_mesh(fe_mesh,
                            quadrature_formula,
                            update_values | update_gradients |
                            update_quadrature_points | update_JxW_values);

    FullMatrix<double> cell_matrix;
    Vector<double>     cell_rhs;

    std::vector<types::global_dof_index> local_dof_indices;

    const FEValuesExtractors::Vector u_fe(0);
    const FEValuesExtractors::Scalar phi_fe(dim);
    const FEValuesExtractors::Scalar phi_fe_free(dim);

    typename DoFHandler<dim>::active_cell_iterator
      cell_m = dof_handler_mesh.begin_active(),
      endc_m = dof_handler_mesh.end();

    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        if(cell_m == endc_m){
          std::string error_msg = std::string(__FILE__) + ":" + std::to_string(__LINE__) + " error here!";
          throw std::runtime_error(error_msg);
        }

        if(cell_is_bulk<dim>(cell))
        {
          fe_values_bulk.reinit(cell);

          const unsigned int dofs_per_cell = fe_bulk.dofs_per_cell;

          cell_matrix.reinit(dofs_per_cell, dofs_per_cell);
          cell_rhs.reinit(dofs_per_cell);

          local_dof_indices.resize(dofs_per_cell);
          cell->get_dof_indices(local_dof_indices);
          const unsigned int n_independent_variables = dofs_per_cell;

          ADHelper           ad_helper(n_independent_variables);

          ad_helper.register_dof_values(current_solution, local_dof_indices);
          const std::vector<ADNumberType> &dof_values_ad =
            ad_helper.get_sensitive_dof_values();
          std::vector<Tensor<2, dim, ADNumberType>> old_solution_u_gradients(
            fe_values_bulk.n_quadrature_points);
          std::vector<Tensor<1, dim, ADNumberType>> old_solution_phi_gradients(
            fe_values_bulk.n_quadrature_points);
          fe_values_bulk[u_fe].get_function_gradients_from_local_dof_values(
            dof_values_ad, old_solution_u_gradients);
          fe_values_bulk[phi_fe].get_function_gradients_from_local_dof_values(
            dof_values_ad, old_solution_phi_gradients);

          ADNumberType energy_ad = ADNumberType(0.0);
          for (const unsigned int q : fe_values_bulk.quadrature_point_indices())
          {
            const double lambda = this->lambda;
            const double mu = this->mu;
            const double c_1 = this->c_1;
            const double c_2 = this->c_2;
            const double epsilon = this->epsilon;
            
            const Tensor<2, dim, ADNumberType> F = Physics::Elasticity::Kinematics::F(old_solution_u_gradients[q]);
            const SymmetricTensor< 2, dim, ADNumberType> C = Physics::Elasticity::Kinematics::C(F);
            const SymmetricTensor< 2, dim, ADNumberType> C_inv = invert(C);
            const SymmetricTensor< 2, dim, ADNumberType> I = unit_symmetric_tensor<dim,ADNumberType>();
            const ADNumberType J = determinant(F);
            const Tensor<1, dim, ADNumberType> E = -old_solution_phi_gradients[q];
            const SymmetricTensor<2, dim, ADNumberType> ExE = symmetrize(outer_product(E,E));
            
            if (J <= 0){
              terminate_loadstep += 1;
              break;
            }
            
            const ADNumberType psi = mu*0.5*(trace(C) - 3)
                                  - mu*std::log(J)
                                  + lambda*0.5*std::pow(std::log(J),2.0)
                                  + c_1*I*ExE
                                  + c_2*C*ExE
                                  - 0.5*epsilon*J*(C_inv*ExE);
                                  
            energy_ad += psi * fe_values_bulk.JxW(q);

          }
          ad_helper.register_energy_functional(energy_ad);
          ad_helper.compute_residual(cell_rhs);
          cell_rhs *= -1.0;
          ad_helper.compute_linearization(cell_matrix);
        } // end if cell is bulk
        else if(cell_is_free<dim>(cell))
        {

          fe_values_free.reinit(cell);
          fe_values_mesh.reinit(cell_m);
          
          std::vector<Tensor<2, dim>> mesh_solution_gradients(
            fe_values_mesh.n_quadrature_points);
          fe_values_mesh[u_fe].get_function_gradients(solution_mesh, 
              mesh_solution_gradients);

          const unsigned int dofs_per_cell = fe_free.dofs_per_cell;

          cell_matrix.reinit(dofs_per_cell, dofs_per_cell);
          cell_rhs.reinit(dofs_per_cell);

          local_dof_indices.resize(dofs_per_cell);
          cell->get_dof_indices(local_dof_indices);
          const unsigned int n_independent_variables = dofs_per_cell;

          ADHelper           ad_helper(n_independent_variables);

          ad_helper.register_dof_values(current_solution, local_dof_indices);
          const std::vector<ADNumberType> &dof_values_ad =
            ad_helper.get_sensitive_dof_values();
          std::vector<Tensor<1, dim, ADNumberType>> old_solution_phi_gradients(
            fe_values_free.n_quadrature_points);
          fe_values_free[phi_fe_free].get_function_gradients_from_local_dof_values(
            dof_values_ad, old_solution_phi_gradients);

          ADNumberType energy_ad = ADNumberType(0.0);
          for (const unsigned int q : fe_values_free.quadrature_point_indices())
          {
            const Tensor<2, dim> F_mesh 
              = Physics::Elasticity::Kinematics::F(mesh_solution_gradients[q]);
            const double J_mesh = determinant(F_mesh);
            const SymmetricTensor< 2, dim> C_mesh 
              = Physics::Elasticity::Kinematics::C(F_mesh);
            
            const ADNumberType epsilon_0 = 8.854e-12;
            
            const Tensor<1, dim, ADNumberType> E = -old_solution_phi_gradients[q];

            const Tensor<2, dim, ADNumberType> ExE = outer_product(E,E); 

            const ADNumberType psi = -0.5*epsilon_0*J_mesh*scalar_product(C_mesh,ExE);
            
            // const ADNumberType psi = -0.5*epsilon_0*scalar_product(E,E);
                                  
            energy_ad += psi * fe_values_free.JxW(q);

          }
          ad_helper.register_energy_functional(energy_ad);
          ad_helper.compute_residual(cell_rhs);
          cell_rhs *= -1.0;
          ad_helper.compute_linearization(cell_matrix);
        } // end if cell is free

        constraints.distribute_local_to_global(
          cell_matrix, cell_rhs, local_dof_indices, system_matrix, residual);

        ++cell_m;
      }
  }

  template <int dim, Differentiation::AD::NumberTypes ADTypeCode>
  void EAPFreeMorphProblem<dim, ADTypeCode>::solve_load_step_NR()
  {

    error_NR.Reset();

    newton_iteration = 0;
    for (; newton_iteration < max_nr_steps; ++newton_iteration) {
      // Reset the tangent matrix and the rhs vector
      system_matrix = 0.0;
      residual = 0.0;
      setup_constraint();
      assemble_system();
      if(terminate_loadstep >= 1) {
        return;
      }

      if (newton_iteration == 0)
        error_NR.Initialize(get_error_residual());

      double error_residual_norm = error_NR.get_normalized_error(get_error_residual());

      std::cout << error_residual_norm << " | ";

      /*Problem has to be solved at least once*/
      if (newton_iteration > 0 && error_residual_norm <= this->nr_tolerance)
        break;
      solve(newton_update);
      //ADD THE NEWTON INCREMENT TO THE LOAD STEP DELTA solution_delta
      solution_delta += newton_update;
    }
  }

  template <int dim, Differentiation::AD::NumberTypes ADTypeCode>
  void EAPFreeMorphProblem<dim, ADTypeCode>::solve(Vector<double> &newton_update_)
  {
    TimerOutput::Scope t(compute_timer, "solve");

    prm.enter_subsection("General");
    const std::string solver_name = prm.get("Solver");
    prm.leave_subsection();

    newton_update_ = 0.0;

    if(solver_name == "umfpack"){
      SparseDirectUMFPACK A_direct;
      A_direct.initialize(system_matrix);
      A_direct.vmult(newton_update_, residual);
    }
    else if(solver_name == "cg"){

      SolverControl            solver_control(1000, 1e-12);
      SolverCG<Vector<double>> cg(solver_control);

      PreconditionSSOR<SparseMatrix<double>> preconditioner;
      preconditioner.initialize(system_matrix, 1.2);

      cg.solve(system_matrix, newton_update_, residual, preconditioner);
    }
    else{
      std::string error_msg = std::string(__FILE__) 
        + ":" + std::to_string(__LINE__) + " wrong solver name!";
      throw std::runtime_error(error_msg);
    }

    constraints.distribute(newton_update_);
  }

  template <int dim, Differentiation::AD::NumberTypes ADTypeCode>
  Vector<double> 
  EAPFreeMorphProblem<dim, ADTypeCode>::get_total_solution(const Vector<double> &solution_delta) const
  {
    Vector<double>  solution_total(solution);
    solution_total += solution_delta;
    return solution_total;
  }

  template <int dim, Differentiation::AD::NumberTypes ADTypeCode>
  double 
  EAPFreeMorphProblem<dim, ADTypeCode>::get_error_residual()
  {
    double res = 0.0;
    for (unsigned int i = 0; i < dof_handler.n_dofs(); ++i) {
      if (!constraints.is_constrained(i)) {
        res += std::pow(residual(i),2.0);
      }
    }

    return res;
  }

  template <int dim, Differentiation::AD::NumberTypes ADTypeCode>
  void EAPFreeMorphProblem<dim, ADTypeCode>::output_results(const unsigned int cycle) const
  {
    TimerOutput::Scope t(compute_timer, "output_results");
    DataOut<dim> data_out;
    data_out.attach_triangulation(triangulation);

    std::vector<std::string> boundary_names(1, "boundary_id"), material_names(1, "material_id"); 

    std::vector<std::string> solution_names(dim, "displacement"), mesh_solution_names(dim, "mesh_sol");
    solution_names.push_back("potential");

    std::vector<DataComponentInterpretation::DataComponentInterpretation>
        multifield_data_component_interpretation(dim, DataComponentInterpretation::component_is_part_of_vector),
        vector_data_component_interpretation(dim, DataComponentInterpretation::component_is_part_of_vector),
        scalar_data_component_interpretation(DataComponentInterpretation::component_is_scalar);
    multifield_data_component_interpretation.push_back(DataComponentInterpretation::component_is_scalar);
    if(cycle != 0){
        data_out.add_data_vector(dof_handler,
                                solution,
                                solution_names,
                                multifield_data_component_interpretation);

        data_out.add_data_vector(dof_handler_mesh,
                                solution_mesh,
                                mesh_solution_names,
                                vector_data_component_interpretation);
    }
    data_out.add_data_vector(tria_boundary_ids,
                            boundary_names);
    data_out.add_data_vector(tria_material_ids,
                            material_names);
    data_out.build_patches();

    std::string output_file_prefix;

    prm.enter_subsection("Output filename");
    output_file_prefix = prm.get("Output filename");
    prm.leave_subsection();

    std::ofstream output(output_file_prefix + "-" + std::to_string(cycle) + ".vtk");
    data_out.write_vtk(output);
  }


  template <int dim, Differentiation::AD::NumberTypes ADTypeCode>
  void EAPFreeMorphProblem<dim, ADTypeCode>::make_grid()
  {
    TimerOutput::Scope t(compute_timer, "make_grid");
    prm.enter_subsection("Mesh & geometry parameters");
    const std::string geometry_type = prm.get("Geometry type");
    prm.leave_subsection();

    if(geometry_type == "plate_with_a_hole")
      plate_with_a_hole<dim>(prm, triangulation, tria_boundary_ids, tria_material_ids);
    else if(geometry_type == "hyper_rectangle")
      hyper_rectangle<dim>(prm, triangulation, tria_boundary_ids, tria_material_ids);
    else if(geometry_type == "hyper_rectangle_hole")
      hyper_rectangle_hole<dim>(prm, triangulation, tria_boundary_ids, tria_material_ids);
    else {
      std::string error_msg = std::string(__FILE__) + ":" + std::to_string(__LINE__) + " wrong geometry type!";
      throw std::runtime_error(error_msg);
    }
  }

  template <int dim, Differentiation::AD::NumberTypes ADTypeCode>
  void EAPFreeMorphProblem<dim, ADTypeCode>::run()
  {
    init_load = 1.0 / number_load_steps;

    accum_load = 0.0;
    load_step = 0;
    curr_load = init_load;

    // GridGenerator::hyper_cube(triangulation, -1, 1, true);
    // triangulation.refine_global(4);
    make_grid();

    setup_system();
    output_results(load_step);
    ++load_step;

    while (true) {

      solution_delta = 0.0;

      std::cout << "LS: " << load_step << " : ";
      solve_load_step_NR();
      std::cout << "terminate_loadstep: " << terminate_loadstep << " : " << std::endl;
      if (newton_iteration >= max_nr_steps || terminate_loadstep >= 1) {
        curr_load *= 0.5;
        terminate_loadstep = 0;
        continue;
      }
      accum_load += curr_load;

      solution += solution_delta;
      setup_constraint_mesh();
      assemble_system_mesh();
      solve_mesh();
      output_results(load_step);
      ++load_step;
      if(accum_load >= 1.0)
        break;

      if (newton_iteration <= 5)
        curr_load *= 2.;

      init_load = curr_load;
      if(init_load > 1.)
        init_load = 1.;

      if (curr_load + accum_load > 1.)
        curr_load = 1. - accum_load;
    } // end loop over time steps

  }
} // namespace AD_EAP_FreeMorph


int main(int argc, char *argv[])
{

  try
    {
      using namespace dealii;
      using namespace AD_EAP_FreeMorph;

      std::string parameter_file;
      ParameterHandler prm;
      ParameterReader  param(prm);
      if (argc == 1) {
        param.declare_parameters();
        param.output_default_parameters();
        param.read_parameters("input.prm");
      }else if (argc == 2){
        parameter_file = argv[1];
        param.read_parameters(parameter_file);
      }else
      throw std::runtime_error ("Wrong number of parameters");

      prm.enter_subsection("General");
      const unsigned int dim = prm.get_integer("Problem dimension");
      prm.leave_subsection();

      constexpr Differentiation::AD::NumberTypes ADTypeCode =
            Differentiation::AD::NumberTypes::sacado_dfad_dfad;
      if(dim == 2){
        AD_EAP_FreeMorph::EAPFreeMorphProblem<2, ADTypeCode> elastic_problem_2d(prm);
        elastic_problem_2d.run();
      } else if(dim == 3){
        AD_EAP_FreeMorph::EAPFreeMorphProblem<3, ADTypeCode> elastic_problem_3d(prm);
        elastic_problem_3d.run();
      }
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;

      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }

  return 0;
}
