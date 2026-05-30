import math

def simSetup(body1Mass, body2Mass):
    mu = body2Mass / (body1Mass + body2Mass)
    body1XPosition = -mu
    body2XPosition = 1 - mu
    return mu, body1XPosition, body2XPosition


def particleToBody1And2Distance(particleXPos, particleYPos, mu):
    r1 = math.sqrt((particleXPos + mu)**2 + particleYPos**2)
    r2 = math.sqrt((particleXPos + mu - 1)**2 + particleYPos**2)
    return r1, r2


def accelerationX(particleYVel, particleXPos, mu, body1Dist, body2Dist):
    return (2*particleYVel + particleXPos
            - (1-mu)*(particleXPos+mu)   / body1Dist**3
            - mu    *(particleXPos-1+mu) / body2Dist**3)


def accelerationY(particleXVel, particleYPos, mu, body1Dist, body2Dist):
    return (-2*particleXVel + particleYPos
            - (1-mu)*particleYPos / body1Dist**3
            - mu    *particleYPos / body2Dist**3)


def _derivatives(x, y, vx, vy, mu):
    """Return (dx/dt, dy/dt, dvx/dt, dvy/dt)."""
    r1, r2 = particleToBody1And2Distance(x, y, mu)
    ax = accelerationX(vy, x, mu, r1, r2)
    ay = accelerationY(vx, y, mu, r1, r2)
    return vx, vy, ax, ay


def simulationStep(particleState, timestep, mu):
    x, y, vx, vy = particleState
    dt = timestep

    dx1, dy1, dvx1, dvy1 = _derivatives(x,           y,           vx,           vy,           mu)
    dx2, dy2, dvx2, dvy2 = _derivatives(x+dt/2*dx1,  y+dt/2*dy1,  vx+dt/2*dvx1, vy+dt/2*dvy1, mu)
    dx3, dy3, dvx3, dvy3 = _derivatives(x+dt/2*dx2,  y+dt/2*dy2,  vx+dt/2*dvx2, vy+dt/2*dvy2, mu)
    dx4, dy4, dvx4, dvy4 = _derivatives(x+dt*dx3,    y+dt*dy3,    vx+dt*dvx3,   vy+dt*dvy3,   mu)

    newX  = x  + dt/6*(dx1  + 2*dx2  + 2*dx3  + dx4)
    newY  = y  + dt/6*(dy1  + 2*dy2  + 2*dy3  + dy4)
    newVx = vx + dt/6*(dvx1 + 2*dvx2 + 2*dvx3 + dvx4)
    newVy = vy + dt/6*(dvy1 + 2*dvy2 + 2*dvy3 + dvy4)

    newParticleState = [newX, newY, newVx, newVy]

    # Return acc and distances at the START of the step
    r1, r2 = particleToBody1And2Distance(x, y, mu)
    accX = accelerationX(vy, x, mu, r1, r2)
    accY = accelerationY(vx, y, mu, r1, r2)

    return newParticleState, accX, accY, r1, r2


def generalSimulationLoop(timestepCount, timestep, initialXPos, initialYPos,
                          initialXVel, initialYVel, body1Mass, body2Mass):
    particleState = [initialXPos, initialYPos, initialXVel, initialYVel]
    particleHistory = []

    mu, body1XPosition, body2XPosition = simSetup(body1Mass, body2Mass)

    for i in range(timestepCount + 1):
        time = i * timestep
        r1, r2 = particleToBody1And2Distance(particleState[0], particleState[1], mu)
        accX = accelerationX(particleState[3], particleState[0], mu, r1, r2)
        accY = accelerationY(particleState[2], particleState[1], mu, r1, r2)
        particleHistory.append(particleState.copy() + [accX, accY, r1, r2, time])
        particleState, accX, accY, r1, r2 = simulationStep(particleState, timestep, mu)

    return particleHistory
