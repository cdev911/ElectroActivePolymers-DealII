/* ========================
 * Electro Active Polymers
 * ========================
 * Problem description:
 *   Nonlinear electro-elastostastic solver.
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


namespace AD_EAP_Bulk
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


  template <int dim>
  void hyper_rectangle_hole(ParameterHandler &prm,
                        Triangulation<dim> &triangulation,
                        Vector<double> &tria_boundary_ids)
  {
    prm.enter_subsection("Mesh & geometry parameters");
    const double inner_radius = prm.get_double("Inner radius");
    const double outer_radius = prm.get_double("Outer radius");
    const unsigned int refinement = prm.get_integer("Number of refinements");
    const unsigned int  subdiv_x = prm.get_integer("Number of subdivision in X");
    const unsigned int  subdiv_y = prm.get_integer("Number of subdivision in Y");
    const unsigned int  subdiv_z = prm.get_integer("Number of subdivision in X");
    prm.leave_subsection();


    Triangulation<dim> dummy_tria;
    std::set< typename Triangulation< dim>::active_cell_iterator >	cells_to_remove;
    if(dim == 2)
    {
      std::vector<unsigned int> subdivision = {subdiv_x, subdiv_y};
      Point<dim> p1(-outer_radius,-outer_radius), p2(outer_radius,outer_radius);
      GridGenerator::subdivided_hyper_rectangle(dummy_tria, subdivision, p1, p2, true);
      dummy_tria.refine_global(refinement); 
      
      for (auto cell: dummy_tria.active_cell_iterators()) {
        Point<dim> center = cell->center();
        if(std::abs(center[0]) < inner_radius && std::abs(center[1]) < inner_radius){
          cells_to_remove.insert(cell);
        } 
      } // end of loop over cells
    } // if dim == 2
    else if(dim == 3)
    {
      std::vector<unsigned int> subdivision = {subdiv_x, subdiv_y, subdiv_z};
      Point<dim> p1(-outer_radius,-outer_radius,-outer_radius), p2(outer_radius,outer_radius,outer_radius);
      GridGenerator::subdivided_hyper_rectangle(dummy_tria, subdivision, p1, p2, true);
      dummy_tria.refine_global(refinement); 

      for (auto cell: dummy_tria.active_cell_iterators()) {
        Point<dim> center = cell->center();
        if(std::abs(center[0]) < inner_radius && std::abs(center[1]) < inner_radius ){
          cells_to_remove.insert(cell);
        } 
      } // end of loop over cells
    } // if dim == 3
    
    GridGenerator::create_triangulation_with_removed_cells(dummy_tria, cells_to_remove, triangulation);
      
    tria_boundary_ids.reinit(triangulation.n_active_cells());
    Vector<double> dom_boundary_ids(triangulation.n_active_cells() * GeometryInfo<dim>::faces_per_cell);
    unsigned int counter = 0, cell_counter = 0;
    for (auto cell: triangulation.active_cell_iterators()) {
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
    } // end of loop over cells
  }

    template <int dim>
  void hyper_rectangle(ParameterHandler &prm,
                        Triangulation<dim> &triangulation,
                        Vector<double> &tria_boundary_ids)
  {
    prm.enter_subsection("Mesh & geometry parameters");
    const unsigned int refinement = prm.get_integer("Number of refinements");
    prm.leave_subsection();
    const double bulk = 30e-6;

    if(dim == 2)
    {
      Point<dim> p1(-bulk,-bulk), p2(bulk,bulk);
      GridGenerator::hyper_rectangle(triangulation, p1, p2, true);
      triangulation.refine_global(refinement); 

      tria_boundary_ids.reinit(triangulation.n_active_cells());
      Vector<double> dom_boundary_ids(triangulation.n_active_cells() * GeometryInfo<dim>::faces_per_cell);
      unsigned int counter = 0, cell_counter = 0;
      for (auto cell: triangulation.active_cell_iterators()) {
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
      Point<dim> p1(-bulk,-bulk,-bulk), p2(bulk,bulk,bulk);
      GridGenerator::hyper_rectangle(triangulation, p1, p2, true);
      triangulation.refine_global(refinement); 

      tria_boundary_ids.reinit(triangulation.n_active_cells());
      Vector<double> dom_boundary_ids(triangulation.n_active_cells() * GeometryInfo<dim>::faces_per_cell);
      unsigned int counter = 0, cell_counter = 0;
      for (auto cell: triangulation.active_cell_iterators()) {
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
  void setup_dirichlet_dof_info_hyper_rectangle_edge(ParameterHandler &prm,
                                                DoFHandler<dim> &dof_handler,
                                                std::map<unsigned int, double> &map_dir_dof_index_to_val)
  {
    const double bulk = 30e-6;
    prm.enter_subsection("Boundary conditions");
    const double voltage = prm.get_double("voltage");
    prm.leave_subsection();

    for (auto cell: dof_handler.active_cell_iterators()) {

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

        if(std::abs(std::abs(vertex[0]) - outer_radius) < outer_radius*0.155 && 
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

  template <int dim, Differentiation::AD::NumberTypes ADTypeCode>
  class EAPBulkProblem
  {
  public:
    EAPBulkProblem(ParameterHandler &);
    void run();

  private:

    void make_grid();
    void setup_system();
    void setup_constraint();
    void setup_dirichlet_dof_info();
    void assemble_system();
    void solve(Vector<double> &newton_update);
    void solve_load_step_NR();
    void output_results(const unsigned int cycle) const;
    Vector<double> get_total_solution(const Vector<double> &solution_delta) const;
    double get_error_residual();

    ParameterHandler &prm;

    Triangulation<dim> triangulation;
    DoFHandler<dim>    dof_handler;
    Vector<double> tria_boundary_ids;

    FESystem<dim> fe_bulk;

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

    mutable TimerOutput compute_timer;
  };



  template <int dim, Differentiation::AD::NumberTypes ADTypeCode>
  EAPBulkProblem<dim, ADTypeCode>::EAPBulkProblem(ParameterHandler &prm_)
    : prm(prm_)
    , dof_handler(triangulation)
    , fe_bulk(FE_Q<dim>(1), dim,
         FE_Q<dim>(1), 1)
    , compute_timer(std::cout, TimerOutput::summary, TimerOutput::wall_times)

  {

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
  void EAPBulkProblem<dim, ADTypeCode>::setup_system()
  {
    TimerOutput::Scope t(compute_timer, "setup_system");
    // For the standard problem
    dof_handler.distribute_dofs(fe_bulk);

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

  }

  template <int dim, Differentiation::AD::NumberTypes ADTypeCode>
  void EAPBulkProblem<dim, ADTypeCode>::setup_dirichlet_dof_info()
  {

    prm.enter_subsection("Boundary conditions");
    const std::string boundary_case = prm.get("Boundary conditions case");
    prm.leave_subsection();

    if(boundary_case == "hyper_rectangle_hole")
      setup_dirichlet_dof_info_hyper_rectangle_hole<dim>(prm, dof_handler, map_dir_dof_index_to_val);
    else if(boundary_case == "hyper_rectangle_edge")
      setup_dirichlet_dof_info_hyper_rectangle_edge<dim>(prm, dof_handler, map_dir_dof_index_to_val);
    else
    {
      std::string error_msg = std::string(__FILE__) + ":" + std::to_string(__LINE__) + " wrong boundary case! Specified case is: "
        + boundary_case;
      throw std::runtime_error(error_msg);
    }

  }

  template <int dim, Differentiation::AD::NumberTypes ADTypeCode>
  void EAPBulkProblem<dim, ADTypeCode>::setup_constraint()
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
  void EAPBulkProblem<dim, ADTypeCode>::assemble_system()
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

    const unsigned int dofs_per_cell = fe_bulk.dofs_per_cell;
    
    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double>     cell_rhs(dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    const FEValuesExtractors::Vector u_fe(0);
    const FEValuesExtractors::Scalar phi_fe(dim);


    for (const auto &cell : dof_handler.active_cell_iterators())
    {
      cell_matrix = 0;
      cell_rhs    = 0;

      fe_values_bulk.reinit(cell);

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

      constraints.distribute_local_to_global(
      cell_matrix, cell_rhs, local_dof_indices, system_matrix, residual);

    }
  }

  template <int dim, Differentiation::AD::NumberTypes ADTypeCode>
  void EAPBulkProblem<dim, ADTypeCode>::solve_load_step_NR()
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
  void EAPBulkProblem<dim, ADTypeCode>::solve(Vector<double> &newton_update_)
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

      SolverControl            solver_control(1000, 1e-12*residual.l2_norm());
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
  EAPBulkProblem<dim, ADTypeCode>::get_total_solution(const Vector<double> &solution_delta) const
  {
    Vector<double>  solution_total(solution);
    solution_total += solution_delta;
    return solution_total;
  }

  template <int dim, Differentiation::AD::NumberTypes ADTypeCode>
  double 
  EAPBulkProblem<dim, ADTypeCode>::get_error_residual()
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
  void EAPBulkProblem<dim, ADTypeCode>::output_results(const unsigned int cycle) const
  {
    TimerOutput::Scope t(compute_timer, "output_results");
    DataOut<dim> data_out;
    data_out.attach_triangulation(triangulation);

    std::vector<std::string> boundary_names(1, "boundary_id"); 

    std::vector<std::string> solution_names(dim, "displacement");
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

    }
    data_out.add_data_vector(tria_boundary_ids,
                            boundary_names);

    data_out.build_patches();

    std::string output_file_prefix;

    prm.enter_subsection("Output filename");
    output_file_prefix = prm.get("Output filename");
    prm.leave_subsection();

    std::ofstream output(output_file_prefix + "-" + std::to_string(cycle) + ".vtk");
    data_out.write_vtk(output);
  }

  template <int dim, Differentiation::AD::NumberTypes ADTypeCode>
  void EAPBulkProblem<dim, ADTypeCode>::make_grid()
  {
    TimerOutput::Scope t(compute_timer, "make_grid");
    prm.enter_subsection("Mesh & geometry parameters");
    const std::string geometry_type = prm.get("Geometry type");
    prm.leave_subsection();

    if(geometry_type == "hyper_rectangle_hole")
      hyper_rectangle_hole<dim>(prm, triangulation, tria_boundary_ids);
    else if(geometry_type == "hyper_rectangle")
      hyper_rectangle<dim>(prm, triangulation, tria_boundary_ids);
    else {
      std::string error_msg = std::string(__FILE__) + ":" + std::to_string(__LINE__) + " wrong geometry type!";
      throw std::runtime_error(error_msg);
    }
  }

  template <int dim, Differentiation::AD::NumberTypes ADTypeCode>
  void EAPBulkProblem<dim, ADTypeCode>::run()
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
} // namespace AD_EAP_Bulk


int main(int argc, char *argv[])
{

  try
    {
      using namespace dealii;
      using namespace AD_EAP_Bulk;

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
        AD_EAP_Bulk::EAPBulkProblem<2, ADTypeCode> elastic_problem_2d(prm);
        elastic_problem_2d.run();
      } else if(dim == 3){
        AD_EAP_Bulk::EAPBulkProblem<3, ADTypeCode> elastic_problem_3d(prm);
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
