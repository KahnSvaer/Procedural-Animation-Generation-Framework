# Species Classification for Procedural Animation (WIP)

## Overview

To support procedural animation generation across a wide variety of aquatic species, creatures can be grouped into locomotion-based categories rather than individual species.

A possible initial approach for automatic classification is to create a small dataset and train a lightweight fine-tuned ResNet model to categorize generated creatures into these animation groups. This provides a relatively inexpensive and scalable method for assigning animation templates. As development progresses, additional species and categories can be incorporated.

The goal of this classification is not biological accuracy, but rather the identification of shared motion patterns that can be reused across multiple species.

---

# 1. Fish-Like Swimmers

The most fundamental swimming template and likely the highest-priority category.

### Covers

* Fish
* Sharks
* Dolphins
* Whales
* Tuna
* Some seal-like motion patterns

### Key Properties

* Streamlined body
* Tail-driven propulsion
* Stabilization fins
* Oscillatory swimming motion

### Potential Extensions

* Whiskers (catfish)
* Anglerfish lure systems
* Specialized fin structures
* Species-specific tail profiles

---

# 2. Serpentine Swimmers

Locomotion driven primarily by full-body undulation.

### Covers

* Eels
* Sea snakes
* Lampreys
* Tadpoles

### Key Properties

* Spine-dominant motion
* Continuous wave propagation
* Minimal limb involvement
* Flexible body structure

This category should be relatively straightforward to implement due to its dependence on a single continuous motion chain.

---

# 3. Flat Fin Swimmers

Wide-bodied creatures that rely on fin-wave propulsion.

### Covers

* Rays
* Manta rays

### Key Properties

* Fin-wave locomotion
* Broad flat body
* Gliding behavior
* Reduced tail contribution

This is a comparatively small category but introduces a unique propulsion mechanism distinct from traditional fish swimming.

---

# 4. Flipper Swimmers

Locomotion generated primarily through limb motion.

### Covers

* Sea turtles
* Penguins
* Sea lions
* Frogs (underwater locomotion)

### Key Properties

* Limb-based propulsion
* Relatively rigid torso
* Stroke-driven movement
* Distinct swimming cycles

This category shares several concepts with terrestrial locomotion systems and may benefit from future integration with quadruped animation frameworks.

---

# 5. Cephalopods

Soft-bodied creatures with highly articulated appendages.

### Covers

* Octopus
* Squid
* Cuttlefish

### Key Properties

* Tentacle-based motion
* Soft-body deformation
* Grasping and reaching behaviors
* Hybrid crawling and swimming

### Notes

This is likely one of the most important categories in the project.

A generic swimming animation alone is insufficient. Additional procedural systems may be required for:

* Tentacle coordination
* Object interaction
* Reaching behaviors
* Environmental adaptation
* Soft-body motion synthesis

---

# 6. Gelatinous Organisms

Simple deformable propulsion systems.

### Covers

* Jellyfish
* Comb jellies

### Key Properties

* Rhythmic compression and expansion
* Passive drifting behavior
* Minimal skeletal complexity
* Strong dependence on deformation

This category is relatively small but may require specialized deformation systems.

---

# 7. Arthropods

Segmented creatures with rigid exoskeletons and multiple limbs.

### Covers

* Crabs
* Lobsters
* Shrimp

### Key Properties

* Multi-limb gait systems
* Segmented body structure
* Hard exoskeleton
* Complex appendage coordination
* Claw articulation

### Notes

This category is expected to be among the most technically challenging due to the number of coordinated limbs and movement patterns involved.

---

# Development Priority

Current proposed implementation order:

1. Fish-Like Swimmers
2. Cephalopods
3. Serpentine Swimmers
4. Flat Fin Swimmers
5. Arthropods
6. Gelatinous Organisms
7. Flipper Swimmers

### Rationale

* Fish-like swimmers represent the largest set of reusable aquatic animation patterns.
* Cephalopods introduce important procedural appendage systems.
* Serpentine swimmers should be comparatively easy to implement and validate.
* Flat fin swimmers introduce a new propulsion mechanism while remaining relatively contained.
* Arthropods require significantly more rigging and coordination complexity.
* Gelatinous organisms form a small but unique category.
* Flipper swimmers can potentially leverage future terrestrial locomotion research and therefore can be deferred.

---

# Future Directions

Potential future categories include:

* Amphibious creatures
* Semi-aquatic mammals
* Marine reptiles
* Fantasy creatures
* Hybrid locomotion systems

The classification system should remain extensible so that new categories can be incorporated without requiring major changes to the animation pipeline.
