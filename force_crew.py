from crewai import Agent, Task, Crew

# Gravity Agent
gravity = Agent(
    name="Gravity",
    role="Force responsible for large-scale attraction",
    goal="Apply Newtonian and relativistic gravity to massive bodies.",
    memory=True,
    allow_delegation=True,
)

# Electromagnetic Agent
electromagnetism = Agent(
    name="Electromagnetism",
    role="Force responsible for charge interactions",
    goal="Apply Coulomb’s Law and Maxwell’s equations to charged particles.",
    memory=True,
    allow_delegation=True,
)

# Strong Force Agent
strong_force = Agent(
    name="Strong Force",
    role="Force that binds quarks together",
    goal="Use quantum chromodynamics to ensure quarks remain confined within protons and neutrons.",
    memory=True,
    allow_delegation=True,
)

# Weak Force Agent
weak_force = Agent(
    name="Weak Force",
    role="Force responsible for nuclear decay",
    goal="Facilitate particle decay via weak interactions, including W/Z boson exchange.",
    memory=True,
    allow_delegation=True,
)

# Define Tasks for each force
gravity_task = Task(description="Simulate the gravitational influence on celestial bodies.", agent=gravity)
electromagnetism_task = Task(description="Calculate the forces between charged particles.", agent=electromagnetism)
strong_force_task = Task(description="Ensure quark confinement within nucleons.", agent=strong_force)
weak_force_task = Task(description="Facilitate beta decay in unstable nuclei.", agent=weak_force)

# Assemble the Crew
force_crew = Crew(
    agents=[gravity, electromagnetism, strong_force, weak_force],
    tasks=[gravity_task, electromagnetism_task, strong_force_task, weak_force_task]
)

# Run the simulation
force_crew.kickoff()
