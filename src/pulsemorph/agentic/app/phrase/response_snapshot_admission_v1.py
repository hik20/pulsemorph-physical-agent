"""Root-reviewed digest admission; populated only after the explicit full import.

The compact store cannot grant itself admission by rewriting its own metadata.
A future dataset needs a separately reviewed admission version.
"""

ADMISSION_SHA256 = '0271d1444bdf476d0523d7680d40908f491106fcc6946394b0bd75989278ec9b'
