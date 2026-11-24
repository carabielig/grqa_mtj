# tests

The tests can be run with `python -m unittest`.

### test_grqa

This Python package contains unittests for several parts of the code. The main purpose is to make sure 
that all versions of the individual approaches (slow and fast cpu approach, gpu approach) lead to 
the same results.

#### test_line_sampling

Test all versions of the **line sampling and patch sampling** approach using an artificial recurrence plot 
(with known hyper-surfaces), a square image and a rectangular image. Also contains tests for the 
Recurrence Quantification Analysis measures based on the line length/patch size distributions using a 
predictable image, an unpredictable image (with known expected values) and a comparison between 
a predictable and unpredictabe image (expectation: 
measures for predictable image > measures for unpredictable image).

#### test_mass_distribution

Test the mass distribution for different maximum box sizes, as well as the storage of intermediate results and
the lacunarity for a squared and a rectangular image full of 1s (for this the results are known) and
a periodic image. This includes all CPU and GPU approaches for obtaining the mass distribution.