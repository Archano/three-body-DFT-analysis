import math 
def simSetup(body1Mass, body2Mass): 
    mu = (body2Mass)/(body1Mass+body2Mass)
    body1XPosition = -mu 
    body2XPosition = 1 - mu
    return mu, body1XPosition, body2XPosition   

def particleToBody1And2Distance (particleXPos,particleYPos,mu): 
    r1 = math.sqrt(math.pow(particleXPos + mu,2) + math.pow(particleYPos, 2))
    r2 = math.sqrt(math.pow(particleXPos + mu - 1,2) + math.pow(particleYPos, 2))
    return r1, r2

def accelerationX(particleYVel, particleXPos, mu, body1Dist, body2Dist): 
    firstTermGrouping = 2*particleYVel + particleXPos 
    secondTermGrouping = -(1-mu)*(particleXPos+mu)/(math.pow(body1Dist,3))
    thirdTermGrouping = -mu*(particleXPos-1+mu)/(math.pow(body2Dist,3))
    accX = firstTermGrouping + secondTermGrouping + thirdTermGrouping 
    return accX

def accelerationY(particleXVel, particleYPos, mu, body1Dist, body2Dist): 
    firstTermGrouping = -2*particleXVel + particleYPos 
    secondTermGrouping = -(1-mu)*particleYPos/(math.pow(body1Dist,3))
    thirdTermGrouping = -mu*particleYPos/(math.pow(body2Dist,3))
    accY = firstTermGrouping + secondTermGrouping + thirdTermGrouping 
    return accY

def simulationStep(particleState, timestep, mu):

    x = particleState[0]
    y = particleState[1]
    vx = particleState[2]
    vy = particleState[3]

    r1, r2 = particleToBody1And2Distance(x, y, mu)

    accX = accelerationX(vy, x, mu, r1, r2)
    accY = accelerationY(vx, y, mu, r1, r2)

    #Euler update
    newX = x + timestep * vx
    newY = y + timestep * vy
    newVx = vx + timestep * accX
    newVy = vy + timestep * accY

    newParticleState = [newX, newY, newVx, newVy]

    return newParticleState, accX, accY, r1, r2

def generalSimulationLoop(timestepCount, timestep, initialXPos, initialYPos,
                          initialXVel, initialYVel, body1Mass, body2Mass): 
    
    particleState = [initialXPos, initialYPos, initialXVel, initialYVel]
    particleHistory = []

    mu, body1XPosition, body2XPosition = simSetup(body1Mass, body2Mass)

    for i in range(timestepCount + 1):
        time = i * timestep

        r1, r2 = particleToBody1And2Distance(particleState[0],particleState[1],mu)

        accX = accelerationX(particleState[3],particleState[0],mu,r1,r2)

        accY = accelerationY(particleState[2],particleState[1],mu,r1,r2)

        particleHistory.append(particleState.copy() + [accX, accY, r1, r2, time])

        particleState, accX, accY, r1, r2 = simulationStep(particleState,timestep, mu)

    return particleHistory
