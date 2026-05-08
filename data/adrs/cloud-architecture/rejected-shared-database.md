# REJECTED: Single Shared Database for Microservices

## Status
REJECTED

## Context
We are designing a system with 10 microservices, each handling a specific business domain (e.g., Inventory, Orders, Users, Payments). To simplify data management and reduce infrastructure costs, it was proposed that all microservices share a single SQL database.

## Decision
We reject the proposal to use a single shared database for all microservices.

## Consequences of Proposed (Rejected) Approach
1. **Tight Coupling:** Any change to the database schema by one microservice could potentially break all other services. This negates the benefit of independent deployability in a microservices architecture.
2. **Failure Blast Radius:** A single database failure or performance bottleneck would bring down the entire system, creating a single point of failure.
3. **Scalability Bottlenecks:** Different services have different scaling needs. A shared database makes it difficult to scale resources independently based on service demand.
4. **Data Ownership:** It becomes unclear which service "owns" the data, leading to conflicting updates and complex data integrity issues.

## Recommended Alternative
Implement the **Database-per-Service** pattern. Each microservice should have its own private database, ensuring loose coupling, independent scalability, and clear data ownership. Communication between services should happen via APIs or asynchronous messaging.
