solver = pywraplp.Solver('SolveIntegerProblem', pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)

for i in range(58):
    for j in range(58):
        x[i,j]=solver.IntVar(0,1,"x[%i,%i]"%(i,j))
    
z = solver.IntVar(0,200, "z")

objective = solver.Minimize(z)

st_u = dict()
st_d = dict()
for i in range(58):
    for j in range(i,58):
        st_u[i,j] =solver.Add((800-w[i]-w[j])*x[i,j] <= z)
        st_d[i,j] = solver.Add((800-w[i]-w[j])*x[i,j] >= z *-1)
    

for j in range(58):
    solver.Add(solver.Sum([x[i,j] for i in range(58)])==1)
    

for i in range(58):
    solver.Add(solver.Sum([x[i,j] for j in range(58)])==1)
    

for i in range(58):
    for j in range(58):
        solver.Add(x[i,j]==x[j,i])
    

for i in range(58):
    solver.Add(x[i,i]==0)


solver.Solve()