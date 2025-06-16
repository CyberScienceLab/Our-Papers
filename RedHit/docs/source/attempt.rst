garak.attempt
=============

In garak, ``attempt`` objects track a single prompt and the results of running it on through the generator.
Probes work by creating a set of garak.attempt objects and setting their class properties.
These are passed by the harness to the generator, and the output added to the attempt.
Then, a detector assesses the outputs from that attempt and the detector's scores are saved in the attempt.
Finally, an evaluator makes judgments of these scores, and writes hits out to the hitlog for any successful probing attempts.

garak.attempt
=============

.. automodule:: garak.attempt
   :members:
   :undoc-members:
   :show-inheritance:   
