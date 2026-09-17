# Build AI Agents with DSPy[¶](#build-ai-agents-with-dspy)

In this tutorial, we will walk you through how to build an AI agents with DSPy. AI agents refer to the system that can autonomously perceive its environment, make decisions, and take actions to achieve specific goals. Unlike a single model prompt, an agent typically follows a loop of reasoning, planning, and acting, often integrating tools like search engines, APIs, or memory to complete complex tasks.

This tutorial focuses on a popular architecture of AI agents called ReAct, standing for **Re**asoning and **Act**ing, which provides a task description along with a list of tools to LM, then lets LM decide whether to call tools for more obseravations, or generate the final output.

As the demo, let's build a simple airline customer service agent that can do the following:

- Book new trips on behalf of the user.
- Modify existing trips, including flight change and cancellation.
- On tasks it cannot handle, raise a customer support ticket.

We will build it from `dspy.ReAct` module.

## Install Dependencies[¶](#install-dependencies)

Before starting, let's install the required packages:

```
!pip install -qU dspy pydantic
```

Recommended: Set up MLflow Tracing to understand what's happening under the hood.

### MLflow DSPy Integration[¶](#mlflow-dspy-integration)

[MLflow](https://mlflow.org/) is an LLMOps tool that natively integrates with DSPy and offer explainability and experiment tracking. In this tutorial, you can use MLflow to visualize prompts and optimization progress as traces to understand the DSPy's behavior better. You can set up MLflow easily by following the four steps below.

1. Install MLflow

```
%pip install mlflow>=3.0.0
```

2. Start MLflow UI in a separate terminal

```
mlflow ui --port 5000 --backend-store-uri sqlite:///mlruns.db
```

3. Connect the notebook to MLflow

```
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("DSPy")
```

4. Enabling tracing.

```
mlflow.dspy.autolog()
```

To learn more about the integration, visit [MLflow DSPy Documentation](https://mlflow.org/docs/latest/llms/dspy/index.html) as well.

## Define Tools[¶](#define-tools)

We need to prepare a list of tools so that the agent can behave like a human airline service agent:

- `fetch_flight_info`: get flight information for certain dates.
- `pick_flight`: pick the best flight based on some criteria.
- `book_flight`: book a flight on behalf of the user.
- `fetch_itinerary`: get the information of a booked itinerary.
- `cancel_itinerary`: cancel a booked itinerary.
- `get_user_info`: get users' information.
- `file_ticket`: file a backlog ticket to have human assist.

### Define Data Structure[¶](#define-data-structure)

Before defining the tools, we need to define the data structure. In real production, this will be the database schema. As a demo, we just define the data structure as [pydantic models](https://docs.pydantic.dev/latest/concepts/models/) for simplicity.

In \[1\]:

Copied!

```
from pydantic import BaseModel

class Date(BaseModel):
    # Somehow LLM is bad at specifying `datetime.datetime`, so
    # we define a custom class to represent the date.
    year: int
    month: int
    day: int
    hour: int

class UserProfile(BaseModel):
    user_id: str
    name: str
    email: str

class Flight(BaseModel):
    flight_id: str
    date_time: Date
    origin: str
    destination: str
    duration: float
    price: float

class Itinerary(BaseModel):
    confirmation_number: str
    user_profile: UserProfile
    flight: Flight

class Ticket(BaseModel):
    user_request: str
    user_profile: UserProfile
```

from pydantic import BaseModel class Date(BaseModel):

# Somehow LLM is bad at specifying `datetime.datetime`, so

# we define a custom class to represent the date.

year: int month: int day: int hour: int class UserProfile(BaseModel): user_id: str name: str email: str class Flight(BaseModel): flight_id: str date_time: Date origin: str destination: str duration: float price: float class Itinerary(BaseModel): confirmation_number: str user_profile: UserProfile flight: Flight class Ticket(BaseModel): user_request: str user_profile: UserProfile

### Create Dummy Data[¶](#create-dummy-data)

Let's also create some dummy data so that the airline agent can do the work. We need to create a few flights and a few users, and initialize empty dictionaries for the itineraries and custom support tickets.

In \[2\]:

Copied!

```
user_database = {
    "Adam": UserProfile(user_id="1", name="Adam", email="adam@gmail.com"),
    "Bob": UserProfile(user_id="2", name="Bob", email="bob@gmail.com"),
    "Chelsie": UserProfile(user_id="3", name="Chelsie", email="chelsie@gmail.com"),
    "David": UserProfile(user_id="4", name="David", email="david@gmail.com"),
}

flight_database = {
    "DA123": Flight(
        flight_id="DA123",  # DSPy Airline 123
        origin="SFO",
        destination="JFK",
        date_time=Date(year=2025, month=9, day=1, hour=1),
        duration=3,
        price=200,
    ),
    "DA125": Flight(
        flight_id="DA125",
        origin="SFO",
        destination="JFK",
        date_time=Date(year=2025, month=9, day=1, hour=7),
        duration=9,
        price=500,
    ),
    "DA456": Flight(
        flight_id="DA456",
        origin="SFO",
        destination="SNA",
        date_time=Date(year=2025, month=10, day=1, hour=1),
        duration=2,
        price=100,
    ),
    "DA460": Flight(
        flight_id="DA460",
        origin="SFO",
        destination="SNA",
        date_time=Date(year=2025, month=10, day=1, hour=9),
        duration=2,
        price=120,
    ),
}

itinerary_database = {}
ticket_database = {}
```

user_database = { "Adam": UserProfile(user_id="1", name="Adam", email="adam@gmail.com"), "Bob": UserProfile(user_id="2", name="Bob", email="bob@gmail.com"), "Chelsie": UserProfile(user_id="3", name="Chelsie", email="chelsie@gmail.com"), "David": UserProfile(user_id="4", name="David", email="david@gmail.com"), } flight_database = { "DA123": Flight( flight_id="DA123", # DSPy Airline 123 origin="SFO", destination="JFK", date_time=Date(year=2025, month=9, day=1, hour=1), duration=3, price=200, ), "DA125": Flight( flight_id="DA125", origin="SFO", destination="JFK", date_time=Date(year=2025, month=9, day=1, hour=7), duration=9, price=500, ), "DA456": Flight( flight_id="DA456", origin="SFO", destination="SNA", date_time=Date(year=2025, month=10, day=1, hour=1), duration=2, price=100, ), "DA460": Flight( flight_id="DA460", origin="SFO", destination="SNA", date_time=Date(year=2025, month=10, day=1, hour=9), duration=2, price=120, ), } itinerary_database = {} ticket_database = {}

### Define the Tools[¶](#define-the-tools)

Now we can define the tools. In order to have `dspy.ReAct` function properly, every function should:

- Have a docstring which defines what the tool does. If the function name is self-explanable, then you can leave the docstring empty.
- Have type hint for the arguments, which is necessary for LM to generate the arguments in the right format.

In \[3\]:

Copied!

```
import random
import string


def fetch_flight_info(date: Date, origin: str, destination: str):
    """Fetch flight information from origin to destination on the given date"""
    flights = []

    for flight_id, flight in flight_database.items():
        if (
            flight.date_time.year == date.year
            and flight.date_time.month == date.month
            and flight.date_time.day == date.day
            and flight.origin == origin
            and flight.destination == destination
        ):
            flights.append(flight)
    if len(flights) == 0:
        raise ValueError("No matching flight found!")
    return flights


def fetch_itinerary(confirmation_number: str):
    """Fetch a booked itinerary information from database"""
    return itinerary_database.get(confirmation_number)


def pick_flight(flights: list[Flight]):
    """Pick up the best flight that matches users' request. we pick the shortest, and cheaper one on ties."""
    sorted_flights = sorted(
        flights,
        key=lambda x: (
            x.get("duration") if isinstance(x, dict) else x.duration,
            x.get("price") if isinstance(x, dict) else x.price,
        ),
    )
    return sorted_flights[0]


def _generate_id(length=8):
    chars = string.ascii_lowercase + string.digits
    return "".join(random.choices(chars, k=length))


def book_flight(flight: Flight, user_profile: UserProfile):
    """Book a flight on behalf of the user."""
    confirmation_number = _generate_id()
    while confirmation_number in itinerary_database:
        confirmation_number = _generate_id()
    itinerary_database[confirmation_number] = Itinerary(
        confirmation_number=confirmation_number,
        user_profile=user_profile,
        flight=flight,
    )
    return confirmation_number, itinerary_database[confirmation_number]


def cancel_itinerary(confirmation_number: str, user_profile: UserProfile):
    """Cancel an itinerary on behalf of the user."""
    if confirmation_number in itinerary_database:
        del itinerary_database[confirmation_number]
        return
    raise ValueError("Cannot find the itinerary, please check your confirmation number.")


def get_user_info(name: str):
    """Fetch the user profile from database with given name."""
    return user_database.get(name)


def file_ticket(user_request: str, user_profile: UserProfile):
    """File a customer support ticket if this is something the agent cannot handle."""
    ticket_id = _generate_id(length=6)
    ticket_database[ticket_id] = Ticket(
        user_request=user_request,
        user_profile=user_profile,
    )
    return ticket_id
```

import random import string def fetch_flight_info(date: Date, origin: str, destination: str): """Fetch flight information from origin to destination on the given date""" flights = [] for flight_id, flight in flight_database.items(): if ( flight.date_time.year == date.year and flight.date_time.month == date.month and flight.date_time.day == date.day and flight.origin == origin and flight.destination == destination ): flights.append(flight) if len(flights) == 0: raise ValueError("No matching flight found!") return flights def fetch_itinerary(confirmation_number: str): """Fetch a booked itinerary information from database""" return itinerary_database.get(confirmation_number) def pick_flight(flights: list[Flight]): """Pick up the best flight that matches users' request. we pick the shortest, and cheaper one on ties.""" sorted_flights = sorted( flights, key=lambda x: ( x.get("duration") if isinstance(x, dict) else x.duration, x.get("price") if isinstance(x, dict) else x.price, ), ) return sorted_flights[0] def \_generate_id(length=8): chars = string.ascii_lowercase + string.digits return "".join(random.choices(chars, k=length)) def book_flight(flight: Flight, user_profile: UserProfile): """Book a flight on behalf of the user.""" confirmation_number = \_generate_id() while confirmation_number in itinerary_database: confirmation_number = \_generate_id() itinerary_database[confirmation_number] = Itinerary( confirmation_number=confirmation_number, user_profile=user_profile, flight=flight, ) return confirmation_number, itinerary_database[confirmation_number] def cancel_itinerary(confirmation_number: str, user_profile: UserProfile): """Cancel an itinerary on behalf of the user.""" if confirmation_number in itinerary_database: del itinerary_database[confirmation_number] return raise ValueError("Cannot find the itinerary, please check your confirmation number.") def get_user_info(name: str): """Fetch the user profile from database with given name.""" return user_database.get(name) def file_ticket(user_request: str, user_profile: UserProfile): """File a customer support ticket if this is something the agent cannot handle.""" ticket_id = \_generate_id(length=6) ticket_database[ticket_id] = Ticket( user_request=user_request, user_profile=user_profile, ) return ticket_id

### Create ReAct Agent[¶](#create-react-agent)

Now we can create the ReAct agent via `dspy.ReAct`. We need to provide a signature to `dspy.ReAct` to define task, and the inputs and outputs of the agent, and tell it about the tools it can access.

In \[4\]:

Copied!

```
import dspy

class DSPyAirlineCustomerService(dspy.Signature):
    """You are an airline customer service agent that helps user book and manage flights.

    You are given a list of tools to handle user request, and you should decide the right tool to use in order to
    fulfill users' request."""

    user_request: str = dspy.InputField()
    process_result: str = dspy.OutputField(
        desc=(
                "Message that summarizes the process result, and the information users need, e.g., the "
                "confirmation_number if a new flight is booked."
            )
        )
```

import dspy class DSPyAirlineCustomerService(dspy.Signature): """You are an airline customer service agent that helps user book and manage flights. You are given a list of tools to handle user request, and you should decide the right tool to use in order to fulfill users' request.""" user_request: str = dspy.InputField() process_result: str = dspy.OutputField( desc=( "Message that summarizes the process result, and the information users need, e.g., the " "confirmation_number if a new flight is booked." ) )

In \[5\]:

Copied!

```
agent = dspy.ReAct(
    DSPyAirlineCustomerService,
    tools = [
        fetch_flight_info,
        fetch_itinerary,
        pick_flight,
        book_flight,
        cancel_itinerary,
        get_user_info,
        file_ticket,
    ]
)
```

agent = dspy.ReAct( DSPyAirlineCustomerService, tools = [ fetch_flight_info, fetch_itinerary, pick_flight, book_flight, cancel_itinerary, get_user_info, file_ticket, ] )

## Use the Agent[¶](#use-the-agent)

To interact with the agent, simply provide the request through `user_request`, and the agent will start doing its job.

Select a language model and set up the API keys. We are using gpt-4o-mini here, but you can change to other models. For how to configure the language model, please refer to [this guide](https://dspy.ai/learn/programming/language_models/).

In \[6\]:

Copied!

```
import os

os.environ["OPENAI_API_KEY"] = "{your openai key}"

dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))
```

import os os.environ["OPENAI_API_KEY"] = "{your openai key}" dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

In \[7\]:

Copied!

```
result = agent(user_request="please help me book a flight from SFO to JFK on 09/01/2025, my name is Adam")
print(result)
```

result = agent(user_request="please help me book a flight from SFO to JFK on 09/01/2025, my name is Adam") print(result)

```
Prediction(
    trajectory={'thought_0': 'I need to fetch flight information for the route from SFO to JFK on the specified date of 09/01/2025. Once I have the flight options, I can help Adam pick the best flight and proceed with the booking.', 'tool_name_0': 'fetch_flight_info', 'tool_args_0': {'date': {'year': 2025, 'month': 9, 'day': 1, 'hour': 0}, 'origin': 'SFO', 'destination': 'JFK'}, 'observation_0': [Flight(flight_id='DA123', date_time=Date(year=2025, month=9, day=1, hour=1), origin='SFO', destination='JFK', duration=3.0, price=200.0), Flight(flight_id='DA125', date_time=Date(year=2025, month=9, day=1, hour=7), origin='SFO', destination='JFK', duration=9.0, price=500.0)], 'thought_1': 'I have received the flight options from SFO to JFK on 09/01/2025. There are two flights available: DA123 at 1 AM for $200 and DA125 at 7 AM for $500. I will now pick the best flight, which is the cheaper option, DA123.', 'tool_name_1': 'pick_flight', 'tool_args_1': {'flights': [{'flight_id': 'DA123', 'date_time': {'year': 2025, 'month': 9, 'day': 1, 'hour': 1}, 'origin': 'SFO', 'destination': 'JFK', 'duration': 3.0, 'price': 200.0}, {'flight_id': 'DA125', 'date_time': {'year': 2025, 'month': 9, 'day': 1, 'hour': 7}, 'origin': 'SFO', 'destination': 'JFK', 'duration': 9.0, 'price': 500.0}]}, 'observation_1': Flight(flight_id='DA123', date_time=Date(year=2025, month=9, day=1, hour=1), origin='SFO', destination='JFK', duration=3.0, price=200.0), 'thought_2': "I have successfully picked the best flight, which is DA123. Now, I need to gather Adam's user profile information to proceed with the booking.", 'tool_name_2': 'get_user_info', 'tool_args_2': {'name': 'Adam'}, 'observation_2': UserProfile(user_id='1', name='Adam', email='adam@gmail.com'), 'thought_3': "I have gathered Adam's user profile information. Now I can proceed to book the selected flight DA123 for him.", 'tool_name_3': 'book_flight', 'tool_args_3': {'flight': {'flight_id': 'DA123', 'date_time': {'year': 2025, 'month': 9, 'day': 1, 'hour': 1}, 'origin': 'SFO', 'destination': 'JFK', 'duration': 3.0, 'price': 200.0}, 'user_profile': {'user_id': '1', 'name': 'Adam', 'email': 'adam@gmail.com'}}, 'observation_3': ('94ldyhsx', Itinerary(confirmation_number='94ldyhsx', user_profile=UserProfile(user_id='1', name='Adam', email='adam@gmail.com'), flight=Flight(flight_id='DA123', date_time=Date(year=2025, month=9, day=1, hour=1), origin='SFO', destination='JFK', duration=3.0, price=200.0))), 'thought_4': "Adam's flight has been successfully booked, and I have received the confirmation number. I will now mark the task as complete.", 'tool_name_4': 'finish', 'tool_args_4': {}, 'observation_4': 'Completed.'},
    reasoning="I successfully fetched flight information for the route from SFO to JFK on 09/01/2025. There were two available flights: DA123 at 1 AM for $200 and DA125 at 7 AM for $500. I selected the cheaper option, DA123, and then retrieved Adam's user profile information to proceed with the booking. After booking the flight, I received a confirmation number for the reservation.",
    process_result='Your flight from SFO to JFK on 09/01/2025 has been successfully booked. Your confirmation number is 94ldyhsx.'
)
```

We can see the booked itinerary in the database.

In \[8\]:

Copied!

```
print(itinerary_database)
```

print(itinerary_database)

```
{'94ldyhsx': Itinerary(confirmation_number='94ldyhsx', user_profile=UserProfile(user_id='1', name='Adam', email='adam@gmail.com'), flight=Flight(flight_id='DA123', date_time=Date(year=2025, month=9, day=1, hour=1), origin='SFO', destination='JFK', duration=3.0, price=200.0))}
```

### Interpret the Result[¶](#interpret-the-result)

The result contains the `process_result` as required by the user, and a `reasoning` field that carries the reasoning behind the answer. In addition, it has a `trajectory` field which contains:

- Reasoning (thought) at each step
- Tools picked by LM at each step
- Arguments for tool calling, determined by LM at each step
- Tool execution results at each step

Behind scene, the `dspy.ReAct` is executing a loop, which accumulates tool call information along with the task description, and send to the LM until hits `max_iters` or the LM decides to wrap up. To better interpret the process, let's use `dspy.inspect_history()` to see what's happening inside each step.

In \[9\]:

Copied!

```
dspy.inspect_history(n=10)
```

dspy.inspect_history(n=10)

```
[2025-05-28T01:06:46.819048]

System message:

Your input fields are:
1. `user_request` (str)
2. `trajectory` (str)
Your output fields are:
1. `next_thought` (str)
2. `next_tool_name` (Literal['fetch_flight_info', 'fetch_itinerary', 'pick_flight', 'book_flight', 'cancel_itinerary', 'get_user_info', 'file_ticket', 'finish'])
3. `next_tool_args` (dict[str, Any])
All interactions will be structured in the following way, with the appropriate values filled in.

[[ ## user_request ## ]]
{user_request}

[[ ## trajectory ## ]]
{trajectory}

[[ ## next_thought ## ]]
{next_thought}

[[ ## next_tool_name ## ]]
{next_tool_name}        # note: the value you produce must exactly match (no extra characters) one of: fetch_flight_info; fetch_itinerary; pick_flight; book_flight; cancel_itinerary; get_user_info; file_ticket; finish

[[ ## next_tool_args ## ]]
{next_tool_args}        # note: the value you produce must adhere to the JSON schema: {"type": "object", "additionalProperties": true}

[[ ## completed ## ]]
In adhering to this structure, your objective is: 
        You are an airline customer service agent that helps user book and manage flights. 
        
        You are given a list of tools to handle user request, and you should decide the right tool to use in order to
        fulfill users' request.
        
        You are an Agent. In each episode, you will be given the fields `user_request` as input. And you can see your past trajectory so far.
        Your goal is to use one or more of the supplied tools to collect any necessary information for producing `process_result`.
        
        To do this, you will interleave next_thought, next_tool_name, and next_tool_args in each turn, and also when finishing the task.
        After each tool call, you receive a resulting observation, which gets appended to your trajectory.
        
        When writing next_thought, you may reason about the current situation and plan for future steps.
        When selecting the next_tool_name and its next_tool_args, the tool must be one of:
        
        (1) fetch_flight_info, whose description is <desc>Fetch flight information from origin to destination on the given date</desc>. It takes arguments {'date': {'properties': {'year': {'title': 'Year', 'type': 'integer'}, 'month': {'title': 'Month', 'type': 'integer'}, 'day': {'title': 'Day', 'type': 'integer'}, 'hour': {'title': 'Hour', 'type': 'integer'}}, 'required': ['year', 'month', 'day', 'hour'], 'title': 'Date', 'type': 'object'}, 'origin': {'type': 'string'}, 'destination': {'type': 'string'}} in JSON format.
        (2) fetch_itinerary, whose description is <desc>Fetch a booked itinerary information from database</desc>. It takes arguments {'confirmation_number': {'type': 'string'}} in JSON format.
        (3) pick_flight, whose description is <desc>Pick up the best flight that matches users' request. we pick the shortest, and cheaper one on ties.</desc>. It takes arguments {'flights': {'$defs': {'Date': {'properties': {'year': {'title': 'Year', 'type': 'integer'}, 'month': {'title': 'Month', 'type': 'integer'}, 'day': {'title': 'Day', 'type': 'integer'}, 'hour': {'title': 'Hour', 'type': 'integer'}}, 'required': ['year', 'month', 'day', 'hour'], 'title': 'Date', 'type': 'object'}, 'Flight': {'properties': {'flight_id': {'title': 'Flight Id', 'type': 'string'}, 'date_time': {'$ref': '#/$defs/Date'}, 'origin': {'title': 'Origin', 'type': 'string'}, 'destination': {'title': 'Destination', 'type': 'string'}, 'duration': {'title': 'Duration', 'type': 'number'}, 'price': {'title': 'Price', 'type': 'number'}}, 'required': ['flight_id', 'date_time', 'origin', 'destination', 'duration', 'price'], 'title': 'Flight', 'type': 'object'}}, 'items': {'$ref': '#/$defs/Flight'}, 'type': 'array'}} in JSON format.
        (4) book_flight, whose description is <desc>Book a flight on behalf of the user.</desc>. It takes arguments {'flight': {'properties': {'flight_id': {'title': 'Flight Id', 'type': 'string'}, 'date_time': {'properties': {'year': {'title': 'Year', 'type': 'integer'}, 'month': {'title': 'Month', 'type': 'integer'}, 'day': {'title': 'Day', 'type': 'integer'}, 'hour': {'title': 'Hour', 'type': 'integer'}}, 'required': ['year', 'month', 'day', 'hour'], 'title': 'Date', 'type': 'object'}, 'origin': {'title': 'Origin', 'type': 'string'}, 'destination': {'title': 'Destination', 'type': 'string'}, 'duration': {'title': 'Duration', 'type': 'number'}, 'price': {'title': 'Price', 'type': 'number'}}, 'required': ['flight_id', 'date_time', 'origin', 'destination', 'duration', 'price'], 'title': 'Flight', 'type': 'object'}, 'user_profile': {'properties': {'user_id': {'title': 'User Id', 'type': 'string'}, 'name': {'title': 'Name', 'type': 'string'}, 'email': {'title': 'Email', 'type': 'string'}}, 'required': ['user_id', 'name', 'email'], 'title': 'UserProfile', 'type': 'object'}} in JSON format.
        (5) cancel_itinerary, whose description is <desc>Cancel an itinerary on behalf of the user.</desc>. It takes arguments {'confirmation_number': {'type': 'string'}, 'user_profile': {'properties': {'user_id': {'title': 'User Id', 'type': 'string'}, 'name': {'title': 'Name', 'type': 'string'}, 'email': {'title': 'Email', 'type': 'string'}}, 'required': ['user_id', 'name', 'email'], 'title': 'UserProfile', 'type': 'object'}} in JSON format.
        (6) get_user_info, whose description is <desc>Fetch the user profile from database with given name.</desc>. It takes arguments {'name': {'type': 'string'}} in JSON format.
        (7) file_ticket, whose description is <desc>File a customer support ticket if this is something the agent cannot handle.</desc>. It takes arguments {'user_request': {'type': 'string'}, 'user_profile': {'properties': {'user_id': {'title': 'User Id', 'type': 'string'}, 'name': {'title': 'Name', 'type': 'string'}, 'email': {'title': 'Email', 'type': 'string'}}, 'required': ['user_id', 'name', 'email'], 'title': 'UserProfile', 'type': 'object'}} in JSON format.
        (8) finish, whose description is <desc>Marks the task as complete. That is, signals that all information for producing the outputs, i.e. `process_result`, are now available to be extracted.</desc>. It takes arguments {} in JSON format.


User message:

[[ ## user_request ## ]]
please help me book a flight from SFO to JFK on 09/01/2025, my name is Adam

[[ ## trajectory ## ]]


Respond with the corresponding output fields, starting with the field `[[ ## next_thought ## ]]`, then `[[ ## next_tool_name ## ]]` (must be formatted as a valid Python Literal['fetch_flight_info', 'fetch_itinerary', 'pick_flight', 'book_flight', 'cancel_itinerary', 'get_user_info', 'file_ticket', 'finish']), then `[[ ## next_tool_args ## ]]` (must be formatted as a valid Python dict[str, Any]), and then ending with the marker for `[[ ## completed ## ]]`.


Response:

[[ ## next_thought ## ]]
I need to fetch flight information for the route from SFO to JFK on the specified date of 09/01/2025. Once I have the flight options, I can help Adam pick the best flight and proceed with the booking.

[[ ## next_tool_name ## ]]
fetch_flight_info

[[ ## next_tool_args ## ]]
{"date":{"year":2025,"month":9,"day":1,"hour":0},"origin":"SFO","destination":"JFK"}

[[ ## completed ## ]]





[2025-05-28T01:06:46.832209]

System message:

Your input fields are:
1. `user_request` (str)
2. `trajectory` (str)
Your output fields are:
1. `next_thought` (str)
2. `next_tool_name` (Literal['fetch_flight_info', 'fetch_itinerary', 'pick_flight', 'book_flight', 'cancel_itinerary', 'get_user_info', 'file_ticket', 'finish'])
3. `next_tool_args` (dict[str, Any])
All interactions will be structured in the following way, with the appropriate values filled in.

[[ ## user_request ## ]]
{user_request}

[[ ## trajectory ## ]]
{trajectory}

[[ ## next_thought ## ]]
{next_thought}

[[ ## next_tool_name ## ]]
{next_tool_name}        # note: the value you produce must exactly match (no extra characters) one of: fetch_flight_info; fetch_itinerary; pick_flight; book_flight; cancel_itinerary; get_user_info; file_ticket; finish

[[ ## next_tool_args ## ]]
{next_tool_args}        # note: the value you produce must adhere to the JSON schema: {"type": "object", "additionalProperties": true}

[[ ## completed ## ]]
In adhering to this structure, your objective is: 
        You are an airline customer service agent that helps user book and manage flights. 
        
        You are given a list of tools to handle user request, and you should decide the right tool to use in order to
        fulfill users' request.
        
        You are an Agent. In each episode, you will be given the fields `user_request` as input. And you can see your past trajectory so far.
        Your goal is to use one or more of the supplied tools to collect any necessary information for producing `process_result`.
        
        To do this, you will interleave next_thought, next_tool_name, and next_tool_args in each turn, and also when finishing the task.
        After each tool call, you receive a resulting observation, which gets appended to your trajectory.
        
        When writing next_thought, you may reason about the current situation and plan for future steps.
        When selecting the next_tool_name and its next_tool_args, the tool must be one of:
        
        (1) fetch_flight_info, whose description is <desc>Fetch flight information from origin to destination on the given date</desc>. It takes arguments {'date': {'properties': {'year': {'title': 'Year', 'type': 'integer'}, 'month': {'title': 'Month', 'type': 'integer'}, 'day': {'title': 'Day', 'type': 'integer'}, 'hour': {'title': 'Hour', 'type': 'integer'}}, 'required': ['year', 'month', 'day', 'hour'], 'title': 'Date', 'type': 'object'}, 'origin': {'type': 'string'}, 'destination': {'type': 'string'}} in JSON format.
        (2) fetch_itinerary, whose description is <desc>Fetch a booked itinerary information from database</desc>. It takes arguments {'confirmation_number': {'type': 'string'}} in JSON format.
        (3) pick_flight, whose description is <desc>Pick up the best flight that matches users' request. we pick the shortest, and cheaper one on ties.</desc>. It takes arguments {'flights': {'$defs': {'Date': {'properties': {'year': {'title': 'Year', 'type': 'integer'}, 'month': {'title': 'Month', 'type': 'integer'}, 'day': {'title': 'Day', 'type': 'integer'}, 'hour': {'title': 'Hour', 'type': 'integer'}}, 'required': ['year', 'month', 'day', 'hour'], 'title': 'Date', 'type': 'object'}, 'Flight': {'properties': {'flight_id': {'title': 'Flight Id', 'type': 'string'}, 'date_time': {'$ref': '#/$defs/Date'}, 'origin': {'title': 'Origin', 'type': 'string'}, 'destination': {'title': 'Destination', 'type': 'string'}, 'duration': {'title': 'Duration', 'type': 'number'}, 'price': {'title': 'Price', 'type': 'number'}}, 'required': ['flight_id', 'date_time', 'origin', 'destination', 'duration', 'price'], 'title': 'Flight', 'type': 'object'}}, 'items': {'$ref': '#/$defs/Flight'}, 'type': 'array'}} in JSON format.
        (4) book_flight, whose description is <desc>Book a flight on behalf of the user.</desc>. It takes arguments {'flight': {'properties': {'flight_id': {'title': 'Flight Id', 'type': 'string'}, 'date_time': {'properties': {'year': {'title': 'Year', 'type': 'integer'}, 'month': {'title': 'Month', 'type': 'integer'}, 'day': {'title': 'Day', 'type': 'integer'}, 'hour': {'title': 'Hour', 'type': 'integer'}}, 'required': ['year', 'month', 'day', 'hour'], 'title': 'Date', 'type': 'object'}, 'origin': {'title': 'Origin', 'type': 'string'}, 'destination': {'title': 'Destination', 'type': 'string'}, 'duration': {'title': 'Duration', 'type': 'number'}, 'price': {'title': 'Price', 'type': 'number'}}, 'required': ['flight_id', 'date_time', 'origin', 'destination', 'duration', 'price'], 'title': 'Flight', 'type': 'object'}, 'user_profile': {'properties': {'user_id': {'title': 'User Id', 'type': 'string'}, 'name': {'title': 'Name', 'type': 'string'}, 'email': {'title': 'Email', 'type': 'string'}}, 'required': ['user_id', 'name', 'email'], 'title': 'UserProfile', 'type': 'object'}} in JSON format.
        (5) cancel_itinerary, whose description is <desc>Cancel an itinerary on behalf of the user.</desc>. It takes arguments {'confirmation_number': {'type': 'string'}, 'user_profile': {'properties': {'user_id': {'title': 'User Id', 'type': 'string'}, 'name': {'title': 'Name', 'type': 'string'}, 'email': {'title': 'Email', 'type': 'string'}}, 'required': ['user_id', 'name', 'email'], 'title': 'UserProfile', 'type': 'object'}} in JSON format.
        (6) get_user_info, whose description is <desc>Fetch the user profile from database with given name.</desc>. It takes arguments {'name': {'type': 'string'}} in JSON format.
        (7) file_ticket, whose description is <desc>File a customer support ticket if this is something the agent cannot handle.</desc>. It takes arguments {'user_request': {'type': 'string'}, 'user_profile': {'properties': {'user_id': {'title': 'User Id', 'type': 'string'}, 'name': {'title': 'Name', 'type': 'string'}, 'email': {'title': 'Email', 'type': 'string'}}, 'required': ['user_id', 'name', 'email'], 'title': 'UserProfile', 'type': 'object'}} in JSON format.
        (8) finish, whose description is <desc>Marks the task as complete. That is, signals that all information for producing the outputs, i.e. `process_result`, are now available to be extracted.</desc>. It takes arguments {} in JSON format.


User message:

[[ ## user_request ## ]]
please help me book a flight from SFO to JFK on 09/01/2025, my name is Adam

[[ ## trajectory ## ]]
[[ ## thought_0 ## ]]
I need to fetch flight information for the route from SFO to JFK on the specified date of 09/01/2025. Once I have the flight options, I can help Adam pick the best flight and proceed with the booking.

[[ ## tool_name_0 ## ]]
fetch_flight_info

[[ ## tool_args_0 ## ]]
{"date": {"year": 2025, "month": 9, "day": 1, "hour": 0}, "origin": "SFO", "destination": "JFK"}

[[ ## observation_0 ## ]]
[1] «flight_id='DA123' date_time=Date(year=2025, month=9, day=1, hour=1) origin='SFO' destination='JFK' duration=3.0 price=200.0»
[2] «flight_id='DA125' date_time=Date(year=2025, month=9, day=1, hour=7) origin='SFO' destination='JFK' duration=9.0 price=500.0»

Respond with the corresponding output fields, starting with the field `[[ ## next_thought ## ]]`, then `[[ ## next_tool_name ## ]]` (must be formatted as a valid Python Literal['fetch_flight_info', 'fetch_itinerary', 'pick_flight', 'book_flight', 'cancel_itinerary', 'get_user_info', 'file_ticket', 'finish']), then `[[ ## next_tool_args ## ]]` (must be formatted as a valid Python dict[str, Any]), and then ending with the marker for `[[ ## completed ## ]]`.


Response:

[[ ## next_thought ## ]]
I have received the flight options from SFO to JFK on 09/01/2025. There are two flights available: DA123 at 1 AM for $200 and DA125 at 7 AM for $500. I will now pick the best flight, which is the cheaper option, DA123.

[[ ## next_tool_name ## ]]
pick_flight

[[ ## next_tool_args ## ]]
{"flights":[{"flight_id":"DA123","date_time":{"year":2025,"month":9,"day":1,"hour":1},"origin":"SFO","destination":"JFK","duration":3.0,"price":200.0},{"flight_id":"DA125","date_time":{"year":2025,"month":9,"day":1,"hour":7},"origin":"SFO","destination":"JFK","duration":9.0,"price":500.0}]}

[[ ## completed ## ]]





[2025-05-28T01:06:46.844268]

System message:

Your input fields are:
1. `user_request` (str)
2. `trajectory` (str)
Your output fields are:
1. `next_thought` (str)
2. `next_tool_name` (Literal['fetch_flight_info', 'fetch_itinerary', 'pick_flight', 'book_flight', 'cancel_itinerary', 'get_user_info', 'file_ticket', 'finish'])
3. `next_tool_args` (dict[str, Any])
All interactions will be structured in the following way, with the appropriate values filled in.

[[ ## user_request ## ]]
{user_request}

[[ ## trajectory ## ]]
{trajectory}

[[ ## next_thought ## ]]
{next_thought}

[[ ## next_tool_name ## ]]
{next_tool_name}        # note: the value you produce must exactly match (no extra characters) one of: fetch_flight_info; fetch_itinerary; pick_flight; book_flight; cancel_itinerary; get_user_info; file_ticket; finish

[[ ## next_tool_args ## ]]
{next_tool_args}        # note: the value you produce must adhere to the JSON schema: {"type": "object", "additionalProperties": true}

[[ ## completed ## ]]
In adhering to this structure, your objective is: 
        You are an airline customer service agent that helps user book and manage flights. 
        
        You are given a list of tools to handle user request, and you should decide the right tool to use in order to
        fulfill users' request.
        
        You are an Agent. In each episode, you will be given the fields `user_request` as input. And you can see your past trajectory so far.
        Your goal is to use one or more of the supplied tools to collect any necessary information for producing `process_result`.
        
        To do this, you will interleave next_thought, next_tool_name, and next_tool_args in each turn, and also when finishing the task.
        After each tool call, you receive a resulting observation, which gets appended to your trajectory.
        
        When writing next_thought, you may reason about the current situation and plan for future steps.
        When selecting the next_tool_name and its next_tool_args, the tool must be one of:
        
        (1) fetch_flight_info, whose description is <desc>Fetch flight information from origin to destination on the given date</desc>. It takes arguments {'date': {'properties': {'year': {'title': 'Year', 'type': 'integer'}, 'month': {'title': 'Month', 'type': 'integer'}, 'day': {'title': 'Day', 'type': 'integer'}, 'hour': {'title': 'Hour', 'type': 'integer'}}, 'required': ['year', 'month', 'day', 'hour'], 'title': 'Date', 'type': 'object'}, 'origin': {'type': 'string'}, 'destination': {'type': 'string'}} in JSON format.
        (2) fetch_itinerary, whose description is <desc>Fetch a booked itinerary information from database</desc>. It takes arguments {'confirmation_number': {'type': 'string'}} in JSON format.
        (3) pick_flight, whose description is <desc>Pick up the best flight that matches users' request. we pick the shortest, and cheaper one on ties.</desc>. It takes arguments {'flights': {'$defs': {'Date': {'properties': {'year': {'title': 'Year', 'type': 'integer'}, 'month': {'title': 'Month', 'type': 'integer'}, 'day': {'title': 'Day', 'type': 'integer'}, 'hour': {'title': 'Hour', 'type': 'integer'}}, 'required': ['year', 'month', 'day', 'hour'], 'title': 'Date', 'type': 'object'}, 'Flight': {'properties': {'flight_id': {'title': 'Flight Id', 'type': 'string'}, 'date_time': {'$ref': '#/$defs/Date'}, 'origin': {'title': 'Origin', 'type': 'string'}, 'destination': {'title': 'Destination', 'type': 'string'}, 'duration': {'title': 'Duration', 'type': 'number'}, 'price': {'title': 'Price', 'type': 'number'}}, 'required': ['flight_id', 'date_time', 'origin', 'destination', 'duration', 'price'], 'title': 'Flight', 'type': 'object'}}, 'items': {'$ref': '#/$defs/Flight'}, 'type': 'array'}} in JSON format.
        (4) book_flight, whose description is <desc>Book a flight on behalf of the user.</desc>. It takes arguments {'flight': {'properties': {'flight_id': {'title': 'Flight Id', 'type': 'string'}, 'date_time': {'properties': {'year': {'title': 'Year', 'type': 'integer'}, 'month': {'title': 'Month', 'type': 'integer'}, 'day': {'title': 'Day', 'type': 'integer'}, 'hour': {'title': 'Hour', 'type': 'integer'}}, 'required': ['year', 'month', 'day', 'hour'], 'title': 'Date', 'type': 'object'}, 'origin': {'title': 'Origin', 'type': 'string'}, 'destination': {'title': 'Destination', 'type': 'string'}, 'duration': {'title': 'Duration', 'type': 'number'}, 'price': {'title': 'Price', 'type': 'number'}}, 'required': ['flight_id', 'date_time', 'origin', 'destination', 'duration', 'price'], 'title': 'Flight', 'type': 'object'}, 'user_profile': {'properties': {'user_id': {'title': 'User Id', 'type': 'string'}, 'name': {'title': 'Name', 'type': 'string'}, 'email': {'title': 'Email', 'type': 'string'}}, 'required': ['user_id', 'name', 'email'], 'title': 'UserProfile', 'type': 'object'}} in JSON format.
        (5) cancel_itinerary, whose description is <desc>Cancel an itinerary on behalf of the user.</desc>. It takes arguments {'confirmation_number': {'type': 'string'}, 'user_profile': {'properties': {'user_id': {'title': 'User Id', 'type': 'string'}, 'name': {'title': 'Name', 'type': 'string'}, 'email': {'title': 'Email', 'type': 'string'}}, 'required': ['user_id', 'name', 'email'], 'title': 'UserProfile', 'type': 'object'}} in JSON format.
        (6) get_user_info, whose description is <desc>Fetch the user profile from database with given name.</desc>. It takes arguments {'name': {'type': 'string'}} in JSON format.
        (7) file_ticket, whose description is <desc>File a customer support ticket if this is something the agent cannot handle.</desc>. It takes arguments {'user_request': {'type': 'string'}, 'user_profile': {'properties': {'user_id': {'title': 'User Id', 'type': 'string'}, 'name': {'title': 'Name', 'type': 'string'}, 'email': {'title': 'Email', 'type': 'string'}}, 'required': ['user_id', 'name', 'email'], 'title': 'UserProfile', 'type': 'object'}} in JSON format.
        (8) finish, whose description is <desc>Marks the task as complete. That is, signals that all information for producing the outputs, i.e. `process_result`, are now available to be extracted.</desc>. It takes arguments {} in JSON format.


User message:

[[ ## user_request ## ]]
please help me book a flight from SFO to JFK on 09/01/2025, my name is Adam

[[ ## trajectory ## ]]
[[ ## thought_0 ## ]]
I need to fetch flight information for the route from SFO to JFK on the specified date of 09/01/2025. Once I have the flight options, I can help Adam pick the best flight and proceed with the booking.

[[ ## tool_name_0 ## ]]
fetch_flight_info

[[ ## tool_args_0 ## ]]
{"date": {"year": 2025, "month": 9, "day": 1, "hour": 0}, "origin": "SFO", "destination": "JFK"}

[[ ## observation_0 ## ]]
[1] «flight_id='DA123' date_time=Date(year=2025, month=9, day=1, hour=1) origin='SFO' destination='JFK' duration=3.0 price=200.0»
[2] «flight_id='DA125' date_time=Date(year=2025, month=9, day=1, hour=7) origin='SFO' destination='JFK' duration=9.0 price=500.0»

[[ ## thought_1 ## ]]
I have received the flight options from SFO to JFK on 09/01/2025. There are two flights available: DA123 at 1 AM for $200 and DA125 at 7 AM for $500. I will now pick the best flight, which is the cheaper option, DA123.

[[ ## tool_name_1 ## ]]
pick_flight

[[ ## tool_args_1 ## ]]
{"flights": [{"flight_id": "DA123", "date_time": {"year": 2025, "month": 9, "day": 1, "hour": 1}, "origin": "SFO", "destination": "JFK", "duration": 3.0, "price": 200.0}, {"flight_id": "DA125", "date_time": {"year": 2025, "month": 9, "day": 1, "hour": 7}, "origin": "SFO", "destination": "JFK", "duration": 9.0, "price": 500.0}]}

[[ ## observation_1 ## ]]
{"flight_id": "DA123", "date_time": {"year": 2025, "month": 9, "day": 1, "hour": 1}, "origin": "SFO", "destination": "JFK", "duration": 3.0, "price": 200.0}

Respond with the corresponding output fields, starting with the field `[[ ## next_thought ## ]]`, then `[[ ## next_tool_name ## ]]` (must be formatted as a valid Python Literal['fetch_flight_info', 'fetch_itinerary', 'pick_flight', 'book_flight', 'cancel_itinerary', 'get_user_info', 'file_ticket', 'finish']), then `[[ ## next_tool_args ## ]]` (must be formatted as a valid Python dict[str, Any]), and then ending with the marker for `[[ ## completed ## ]]`.


Response:

[[ ## next_thought ## ]]
I have successfully picked the best flight, which is DA123. Now, I need to gather Adam's user profile information to proceed with the booking.

[[ ## next_tool_name ## ]]
get_user_info

[[ ## next_tool_args ## ]]
{"name": "Adam"}

[[ ## completed ## ]]





[2025-05-28T01:06:46.851917]

System message:

Your input fields are:
1. `user_request` (str)
2. `trajectory` (str)
Your output fields are:
1. `next_thought` (str)
2. `next_tool_name` (Literal['fetch_flight_info', 'fetch_itinerary', 'pick_flight', 'book_flight', 'cancel_itinerary', 'get_user_info', 'file_ticket', 'finish'])
3. `next_tool_args` (dict[str, Any])
All interactions will be structured in the following way, with the appropriate values filled in.

[[ ## user_request ## ]]
{user_request}

[[ ## trajectory ## ]]
{trajectory}

[[ ## next_thought ## ]]
{next_thought}

[[ ## next_tool_name ## ]]
{next_tool_name}        # note: the value you produce must exactly match (no extra characters) one of: fetch_flight_info; fetch_itinerary; pick_flight; book_flight; cancel_itinerary; get_user_info; file_ticket; finish

[[ ## next_tool_args ## ]]
{next_tool_args}        # note: the value you produce must adhere to the JSON schema: {"type": "object", "additionalProperties": true}

[[ ## completed ## ]]
In adhering to this structure, your objective is: 
        You are an airline customer service agent that helps user book and manage flights. 
        
        You are given a list of tools to handle user request, and you should decide the right tool to use in order to
        fulfill users' request.
        
        You are an Agent. In each episode, you will be given the fields `user_request` as input. And you can see your past trajectory so far.
        Your goal is to use one or more of the supplied tools to collect any necessary information for producing `process_result`.
        
        To do this, you will interleave next_thought, next_tool_name, and next_tool_args in each turn, and also when finishing the task.
        After each tool call, you receive a resulting observation, which gets appended to your trajectory.
        
        When writing next_thought, you may reason about the current situation and plan for future steps.
        When selecting the next_tool_name and its next_tool_args, the tool must be one of:
        
        (1) fetch_flight_info, whose description is <desc>Fetch flight information from origin to destination on the given date</desc>. It takes arguments {'date': {'properties': {'year': {'title': 'Year', 'type': 'integer'}, 'month': {'title': 'Month', 'type': 'integer'}, 'day': {'title': 'Day', 'type': 'integer'}, 'hour': {'title': 'Hour', 'type': 'integer'}}, 'required': ['year', 'month', 'day', 'hour'], 'title': 'Date', 'type': 'object'}, 'origin': {'type': 'string'}, 'destination': {'type': 'string'}} in JSON format.
        (2) fetch_itinerary, whose description is <desc>Fetch a booked itinerary information from database</desc>. It takes arguments {'confirmation_number': {'type': 'string'}} in JSON format.
        (3) pick_flight, whose description is <desc>Pick up the best flight that matches users' request. we pick the shortest, and cheaper one on ties.</desc>. It takes arguments {'flights': {'$defs': {'Date': {'properties': {'year': {'title': 'Year', 'type': 'integer'}, 'month': {'title': 'Month', 'type': 'integer'}, 'day': {'title': 'Day', 'type': 'integer'}, 'hour': {'title': 'Hour', 'type': 'integer'}}, 'required': ['year', 'month', 'day', 'hour'], 'title': 'Date', 'type': 'object'}, 'Flight': {'properties': {'flight_id': {'title': 'Flight Id', 'type': 'string'}, 'date_time': {'$ref': '#/$defs/Date'}, 'origin': {'title': 'Origin', 'type': 'string'}, 'destination': {'title': 'Destination', 'type': 'string'}, 'duration': {'title': 'Duration', 'type': 'number'}, 'price': {'title': 'Price', 'type': 'number'}}, 'required': ['flight_id', 'date_time', 'origin', 'destination', 'duration', 'price'], 'title': 'Flight', 'type': 'object'}}, 'items': {'$ref': '#/$defs/Flight'}, 'type': 'array'}} in JSON format.
        (4) book_flight, whose description is <desc>Book a flight on behalf of the user.</desc>. It takes arguments {'flight': {'properties': {'flight_id': {'title': 'Flight Id', 'type': 'string'}, 'date_time': {'properties': {'year': {'title': 'Year', 'type': 'integer'}, 'month': {'title': 'Month', 'type': 'integer'}, 'day': {'title': 'Day', 'type': 'integer'}, 'hour': {'title': 'Hour', 'type': 'integer'}}, 'required': ['year', 'month', 'day', 'hour'], 'title': 'Date', 'type': 'object'}, 'origin': {'title': 'Origin', 'type': 'string'}, 'destination': {'title': 'Destination', 'type': 'string'}, 'duration': {'title': 'Duration', 'type': 'number'}, 'price': {'title': 'Price', 'type': 'number'}}, 'required': ['flight_id', 'date_time', 'origin', 'destination', 'duration', 'price'], 'title': 'Flight', 'type': 'object'}, 'user_profile': {'properties': {'user_id': {'title': 'User Id', 'type': 'string'}, 'name': {'title': 'Name', 'type': 'string'}, 'email': {'title': 'Email', 'type': 'string'}}, 'required': ['user_id', 'name', 'email'], 'title': 'UserProfile', 'type': 'object'}} in JSON format.
        (5) cancel_itinerary, whose description is <desc>Cancel an itinerary on behalf of the user.</desc>. It takes arguments {'confirmation_number': {'type': 'string'}, 'user_profile': {'properties': {'user_id': {'title': 'User Id', 'type': 'string'}, 'name': {'title': 'Name', 'type': 'string'}, 'email': {'title': 'Email', 'type': 'string'}}, 'required': ['user_id', 'name', 'email'], 'title': 'UserProfile', 'type': 'object'}} in JSON format.
        (6) get_user_info, whose description is <desc>Fetch the user profile from database with given name.</desc>. It takes arguments {'name': {'type': 'string'}} in JSON format.
        (7) file_ticket, whose description is <desc>File a customer support ticket if this is something the agent cannot handle.</desc>. It takes arguments {'user_request': {'type': 'string'}, 'user_profile': {'properties': {'user_id': {'title': 'User Id', 'type': 'string'}, 'name': {'title': 'Name', 'type': 'string'}, 'email': {'title': 'Email', 'type': 'string'}}, 'required': ['user_id', 'name', 'email'], 'title': 'UserProfile', 'type': 'object'}} in JSON format.
        (8) finish, whose description is <desc>Marks the task as complete. That is, signals that all information for producing the outputs, i.e. `process_result`, are now available to be extracted.</desc>. It takes arguments {} in JSON format.


User message:

[[ ## user_request ## ]]
please help me book a flight from SFO to JFK on 09/01/2025, my name is Adam

[[ ## trajectory ## ]]
[[ ## thought_0 ## ]]
I need to fetch flight information for the route from SFO to JFK on the specified date of 09/01/2025. Once I have the flight options, I can help Adam pick the best flight and proceed with the booking.

[[ ## tool_name_0 ## ]]
fetch_flight_info

[[ ## tool_args_0 ## ]]
{"date": {"year": 2025, "month": 9, "day": 1, "hour": 0}, "origin": "SFO", "destination": "JFK"}

[[ ## observation_0 ## ]]
[1] «flight_id='DA123' date_time=Date(year=2025, month=9, day=1, hour=1) origin='SFO' destination='JFK' duration=3.0 price=200.0»
[2] «flight_id='DA125' date_time=Date(year=2025, month=9, day=1, hour=7) origin='SFO' destination='JFK' duration=9.0 price=500.0»

[[ ## thought_1 ## ]]
I have received the flight options from SFO to JFK on 09/01/2025. There are two flights available: DA123 at 1 AM for $200 and DA125 at 7 AM for $500. I will now pick the best flight, which is the cheaper option, DA123.

[[ ## tool_name_1 ## ]]
pick_flight

[[ ## tool_args_1 ## ]]
{"flights": [{"flight_id": "DA123", "date_time": {"year": 2025, "month": 9, "day": 1, "hour": 1}, "origin": "SFO", "destination": "JFK", "duration": 3.0, "price": 200.0}, {"flight_id": "DA125", "date_time": {"year": 2025, "month": 9, "day": 1, "hour": 7}, "origin": "SFO", "destination": "JFK", "duration": 9.0, "price": 500.0}]}

[[ ## observation_1 ## ]]
{"flight_id": "DA123", "date_time": {"year": 2025, "month": 9, "day": 1, "hour": 1}, "origin": "SFO", "destination": "JFK", "duration": 3.0, "price": 200.0}

[[ ## thought_2 ## ]]
I have successfully picked the best flight, which is DA123. Now, I need to gather Adam's user profile information to proceed with the booking.

[[ ## tool_name_2 ## ]]
get_user_info

[[ ## tool_args_2 ## ]]
{"name": "Adam"}

[[ ## observation_2 ## ]]
{"user_id": "1", "name": "Adam", "email": "adam@gmail.com"}

Respond with the corresponding output fields, starting with the field `[[ ## next_thought ## ]]`, then `[[ ## next_tool_name ## ]]` (must be formatted as a valid Python Literal['fetch_flight_info', 'fetch_itinerary', 'pick_flight', 'book_flight', 'cancel_itinerary', 'get_user_info', 'file_ticket', 'finish']), then `[[ ## next_tool_args ## ]]` (must be formatted as a valid Python dict[str, Any]), and then ending with the marker for `[[ ## completed ## ]]`.


Response:

[[ ## next_thought ## ]]
I have gathered Adam's user profile information. Now I can proceed to book the selected flight DA123 for him.

[[ ## next_tool_name ## ]]
book_flight

[[ ## next_tool_args ## ]]
{"flight":{"flight_id":"DA123","date_time":{"year":2025,"month":9,"day":1,"hour":1},"origin":"SFO","destination":"JFK","duration":3.0,"price":200.0},"user_profile":{"user_id":"1","name":"Adam","email":"adam@gmail.com"}}

[[ ## completed ## ]]





[2025-05-28T01:06:49.025746]

System message:

Your input fields are:
1. `user_request` (str)
2. `trajectory` (str)
Your output fields are:
1. `next_thought` (str)
2. `next_tool_name` (Literal['fetch_flight_info', 'fetch_itinerary', 'pick_flight', 'book_flight', 'cancel_itinerary', 'get_user_info', 'file_ticket', 'finish'])
3. `next_tool_args` (dict[str, Any])
All interactions will be structured in the following way, with the appropriate values filled in.

[[ ## user_request ## ]]
{user_request}

[[ ## trajectory ## ]]
{trajectory}

[[ ## next_thought ## ]]
{next_thought}

[[ ## next_tool_name ## ]]
{next_tool_name}        # note: the value you produce must exactly match (no extra characters) one of: fetch_flight_info; fetch_itinerary; pick_flight; book_flight; cancel_itinerary; get_user_info; file_ticket; finish

[[ ## next_tool_args ## ]]
{next_tool_args}        # note: the value you produce must adhere to the JSON schema: {"type": "object", "additionalProperties": true}

[[ ## completed ## ]]
In adhering to this structure, your objective is: 
        You are an airline customer service agent that helps user book and manage flights. 
        
        You are given a list of tools to handle user request, and you should decide the right tool to use in order to
        fulfill users' request.
        
        You are an Agent. In each episode, you will be given the fields `user_request` as input. And you can see your past trajectory so far.
        Your goal is to use one or more of the supplied tools to collect any necessary information for producing `process_result`.
        
        To do this, you will interleave next_thought, next_tool_name, and next_tool_args in each turn, and also when finishing the task.
        After each tool call, you receive a resulting observation, which gets appended to your trajectory.
        
        When writing next_thought, you may reason about the current situation and plan for future steps.
        When selecting the next_tool_name and its next_tool_args, the tool must be one of:
        
        (1) fetch_flight_info, whose description is <desc>Fetch flight information from origin to destination on the given date</desc>. It takes arguments {'date': {'properties': {'year': {'title': 'Year', 'type': 'integer'}, 'month': {'title': 'Month', 'type': 'integer'}, 'day': {'title': 'Day', 'type': 'integer'}, 'hour': {'title': 'Hour', 'type': 'integer'}}, 'required': ['year', 'month', 'day', 'hour'], 'title': 'Date', 'type': 'object'}, 'origin': {'type': 'string'}, 'destination': {'type': 'string'}} in JSON format.
        (2) fetch_itinerary, whose description is <desc>Fetch a booked itinerary information from database</desc>. It takes arguments {'confirmation_number': {'type': 'string'}} in JSON format.
        (3) pick_flight, whose description is <desc>Pick up the best flight that matches users' request. we pick the shortest, and cheaper one on ties.</desc>. It takes arguments {'flights': {'$defs': {'Date': {'properties': {'year': {'title': 'Year', 'type': 'integer'}, 'month': {'title': 'Month', 'type': 'integer'}, 'day': {'title': 'Day', 'type': 'integer'}, 'hour': {'title': 'Hour', 'type': 'integer'}}, 'required': ['year', 'month', 'day', 'hour'], 'title': 'Date', 'type': 'object'}, 'Flight': {'properties': {'flight_id': {'title': 'Flight Id', 'type': 'string'}, 'date_time': {'$ref': '#/$defs/Date'}, 'origin': {'title': 'Origin', 'type': 'string'}, 'destination': {'title': 'Destination', 'type': 'string'}, 'duration': {'title': 'Duration', 'type': 'number'}, 'price': {'title': 'Price', 'type': 'number'}}, 'required': ['flight_id', 'date_time', 'origin', 'destination', 'duration', 'price'], 'title': 'Flight', 'type': 'object'}}, 'items': {'$ref': '#/$defs/Flight'}, 'type': 'array'}} in JSON format.
        (4) book_flight, whose description is <desc>Book a flight on behalf of the user.</desc>. It takes arguments {'flight': {'properties': {'flight_id': {'title': 'Flight Id', 'type': 'string'}, 'date_time': {'properties': {'year': {'title': 'Year', 'type': 'integer'}, 'month': {'title': 'Month', 'type': 'integer'}, 'day': {'title': 'Day', 'type': 'integer'}, 'hour': {'title': 'Hour', 'type': 'integer'}}, 'required': ['year', 'month', 'day', 'hour'], 'title': 'Date', 'type': 'object'}, 'origin': {'title': 'Origin', 'type': 'string'}, 'destination': {'title': 'Destination', 'type': 'string'}, 'duration': {'title': 'Duration', 'type': 'number'}, 'price': {'title': 'Price', 'type': 'number'}}, 'required': ['flight_id', 'date_time', 'origin', 'destination', 'duration', 'price'], 'title': 'Flight', 'type': 'object'}, 'user_profile': {'properties': {'user_id': {'title': 'User Id', 'type': 'string'}, 'name': {'title': 'Name', 'type': 'string'}, 'email': {'title': 'Email', 'type': 'string'}}, 'required': ['user_id', 'name', 'email'], 'title': 'UserProfile', 'type': 'object'}} in JSON format.
        (5) cancel_itinerary, whose description is <desc>Cancel an itinerary on behalf of the user.</desc>. It takes arguments {'confirmation_number': {'type': 'string'}, 'user_profile': {'properties': {'user_id': {'title': 'User Id', 'type': 'string'}, 'name': {'title': 'Name', 'type': 'string'}, 'email': {'title': 'Email', 'type': 'string'}}, 'required': ['user_id', 'name', 'email'], 'title': 'UserProfile', 'type': 'object'}} in JSON format.
        (6) get_user_info, whose description is <desc>Fetch the user profile from database with given name.</desc>. It takes arguments {'name': {'type': 'string'}} in JSON format.
        (7) file_ticket, whose description is <desc>File a customer support ticket if this is something the agent cannot handle.</desc>. It takes arguments {'user_request': {'type': 'string'}, 'user_profile': {'properties': {'user_id': {'title': 'User Id', 'type': 'string'}, 'name': {'title': 'Name', 'type': 'string'}, 'email': {'title': 'Email', 'type': 'string'}}, 'required': ['user_id', 'name', 'email'], 'title': 'UserProfile', 'type': 'object'}} in JSON format.
        (8) finish, whose description is <desc>Marks the task as complete. That is, signals that all information for producing the outputs, i.e. `process_result`, are now available to be extracted.</desc>. It takes arguments {} in JSON format.


User message:

[[ ## user_request ## ]]
please help me book a flight from SFO to JFK on 09/01/2025, my name is Adam

[[ ## trajectory ## ]]
[[ ## thought_0 ## ]]
I need to fetch flight information for the route from SFO to JFK on the specified date of 09/01/2025. Once I have the flight options, I can help Adam pick the best flight and proceed with the booking.

[[ ## tool_name_0 ## ]]
fetch_flight_info

[[ ## tool_args_0 ## ]]
{"date": {"year": 2025, "month": 9, "day": 1, "hour": 0}, "origin": "SFO", "destination": "JFK"}

[[ ## observation_0 ## ]]
[1] «flight_id='DA123' date_time=Date(year=2025, month=9, day=1, hour=1) origin='SFO' destination='JFK' duration=3.0 price=200.0»
[2] «flight_id='DA125' date_time=Date(year=2025, month=9, day=1, hour=7) origin='SFO' destination='JFK' duration=9.0 price=500.0»

[[ ## thought_1 ## ]]
I have received the flight options from SFO to JFK on 09/01/2025. There are two flights available: DA123 at 1 AM for $200 and DA125 at 7 AM for $500. I will now pick the best flight, which is the cheaper option, DA123.

[[ ## tool_name_1 ## ]]
pick_flight

[[ ## tool_args_1 ## ]]
{"flights": [{"flight_id": "DA123", "date_time": {"year": 2025, "month": 9, "day": 1, "hour": 1}, "origin": "SFO", "destination": "JFK", "duration": 3.0, "price": 200.0}, {"flight_id": "DA125", "date_time": {"year": 2025, "month": 9, "day": 1, "hour": 7}, "origin": "SFO", "destination": "JFK", "duration": 9.0, "price": 500.0}]}

[[ ## observation_1 ## ]]
{"flight_id": "DA123", "date_time": {"year": 2025, "month": 9, "day": 1, "hour": 1}, "origin": "SFO", "destination": "JFK", "duration": 3.0, "price": 200.0}

[[ ## thought_2 ## ]]
I have successfully picked the best flight, which is DA123. Now, I need to gather Adam's user profile information to proceed with the booking.

[[ ## tool_name_2 ## ]]
get_user_info

[[ ## tool_args_2 ## ]]
{"name": "Adam"}

[[ ## observation_2 ## ]]
{"user_id": "1", "name": "Adam", "email": "adam@gmail.com"}

[[ ## thought_3 ## ]]
I have gathered Adam's user profile information. Now I can proceed to book the selected flight DA123 for him.

[[ ## tool_name_3 ## ]]
book_flight

[[ ## tool_args_3 ## ]]
{"flight": {"flight_id": "DA123", "date_time": {"year": 2025, "month": 9, "day": 1, "hour": 1}, "origin": "SFO", "destination": "JFK", "duration": 3.0, "price": 200.0}, "user_profile": {"user_id": "1", "name": "Adam", "email": "adam@gmail.com"}}

[[ ## observation_3 ## ]]
["94ldyhsx", {"confirmation_number": "94ldyhsx", "user_profile": {"user_id": "1", "name": "Adam", "email": "adam@gmail.com"}, "flight": {"flight_id": "DA123", "date_time": {"year": 2025, "month": 9, "day": 1, "hour": 1}, "origin": "SFO", "destination": "JFK", "duration": 3.0, "price": 200.0}}]

Respond with the corresponding output fields, starting with the field `[[ ## next_thought ## ]]`, then `[[ ## next_tool_name ## ]]` (must be formatted as a valid Python Literal['fetch_flight_info', 'fetch_itinerary', 'pick_flight', 'book_flight', 'cancel_itinerary', 'get_user_info', 'file_ticket', 'finish']), then `[[ ## next_tool_args ## ]]` (must be formatted as a valid Python dict[str, Any]), and then ending with the marker for `[[ ## completed ## ]]`.


Response:

[[ ## next_thought ## ]]
Adam's flight has been successfully booked, and I have received the confirmation number. I will now mark the task as complete.

[[ ## next_tool_name ## ]]
finish

[[ ## next_tool_args ## ]]
{}

[[ ## completed ## ]]





[2025-05-28T01:06:51.599439]

System message:

Your input fields are:
1. `user_request` (str)
2. `trajectory` (str)
Your output fields are:
1. `reasoning` (str)
2. `process_result` (str): Message that summarizes the process result, and the information users need, e.g., the confirmation_number if a new flight is booked.
All interactions will be structured in the following way, with the appropriate values filled in.

[[ ## user_request ## ]]
{user_request}

[[ ## trajectory ## ]]
{trajectory}

[[ ## reasoning ## ]]
{reasoning}

[[ ## process_result ## ]]
{process_result}

[[ ## completed ## ]]
In adhering to this structure, your objective is: 
        You are an airline customer service agent that helps user book and manage flights. 
        
        You are given a list of tools to handle user request, and you should decide the right tool to use in order to
        fulfill users' request.


User message:

[[ ## user_request ## ]]
please help me book a flight from SFO to JFK on 09/01/2025, my name is Adam

[[ ## trajectory ## ]]
[[ ## thought_0 ## ]]
I need to fetch flight information for the route from SFO to JFK on the specified date of 09/01/2025. Once I have the flight options, I can help Adam pick the best flight and proceed with the booking.

[[ ## tool_name_0 ## ]]
fetch_flight_info

[[ ## tool_args_0 ## ]]
{"date": {"year": 2025, "month": 9, "day": 1, "hour": 0}, "origin": "SFO", "destination": "JFK"}

[[ ## observation_0 ## ]]
[1] «flight_id='DA123' date_time=Date(year=2025, month=9, day=1, hour=1) origin='SFO' destination='JFK' duration=3.0 price=200.0»
[2] «flight_id='DA125' date_time=Date(year=2025, month=9, day=1, hour=7) origin='SFO' destination='JFK' duration=9.0 price=500.0»

[[ ## thought_1 ## ]]
I have received the flight options from SFO to JFK on 09/01/2025. There are two flights available: DA123 at 1 AM for $200 and DA125 at 7 AM for $500. I will now pick the best flight, which is the cheaper option, DA123.

[[ ## tool_name_1 ## ]]
pick_flight

[[ ## tool_args_1 ## ]]
{"flights": [{"flight_id": "DA123", "date_time": {"year": 2025, "month": 9, "day": 1, "hour": 1}, "origin": "SFO", "destination": "JFK", "duration": 3.0, "price": 200.0}, {"flight_id": "DA125", "date_time": {"year": 2025, "month": 9, "day": 1, "hour": 7}, "origin": "SFO", "destination": "JFK", "duration": 9.0, "price": 500.0}]}

[[ ## observation_1 ## ]]
{"flight_id": "DA123", "date_time": {"year": 2025, "month": 9, "day": 1, "hour": 1}, "origin": "SFO", "destination": "JFK", "duration": 3.0, "price": 200.0}

[[ ## thought_2 ## ]]
I have successfully picked the best flight, which is DA123. Now, I need to gather Adam's user profile information to proceed with the booking.

[[ ## tool_name_2 ## ]]
get_user_info

[[ ## tool_args_2 ## ]]
{"name": "Adam"}

[[ ## observation_2 ## ]]
{"user_id": "1", "name": "Adam", "email": "adam@gmail.com"}

[[ ## thought_3 ## ]]
I have gathered Adam's user profile information. Now I can proceed to book the selected flight DA123 for him.

[[ ## tool_name_3 ## ]]
book_flight

[[ ## tool_args_3 ## ]]
{"flight": {"flight_id": "DA123", "date_time": {"year": 2025, "month": 9, "day": 1, "hour": 1}, "origin": "SFO", "destination": "JFK", "duration": 3.0, "price": 200.0}, "user_profile": {"user_id": "1", "name": "Adam", "email": "adam@gmail.com"}}

[[ ## observation_3 ## ]]
["94ldyhsx", {"confirmation_number": "94ldyhsx", "user_profile": {"user_id": "1", "name": "Adam", "email": "adam@gmail.com"}, "flight": {"flight_id": "DA123", "date_time": {"year": 2025, "month": 9, "day": 1, "hour": 1}, "origin": "SFO", "destination": "JFK", "duration": 3.0, "price": 200.0}}]

[[ ## thought_4 ## ]]
Adam's flight has been successfully booked, and I have received the confirmation number. I will now mark the task as complete.

[[ ## tool_name_4 ## ]]
finish

[[ ## tool_args_4 ## ]]
{}

[[ ## observation_4 ## ]]
Completed.

Respond with the corresponding output fields, starting with the field `[[ ## reasoning ## ]]`, then `[[ ## process_result ## ]]`, and then ending with the marker for `[[ ## completed ## ]]`.


Response:

[[ ## reasoning ## ]]
I successfully fetched flight information for the route from SFO to JFK on 09/01/2025. There were two available flights: DA123 at 1 AM for $200 and DA125 at 7 AM for $500. I selected the cheaper option, DA123, and then retrieved Adam's user profile information to proceed with the booking. After booking the flight, I received a confirmation number for the reservation.

[[ ## process_result ## ]]
Your flight from SFO to JFK on 09/01/2025 has been successfully booked. Your confirmation number is 94ldyhsx.

[[ ## completed ## ]]
```

We can see that in each LM call, the user message includes the information of previous tool calls, along with the task description.

Let's try a different task.

In \[10\]:

Copied!

```
confirmation_number = "{copy the confirmation number here}"

result = agent(user_request=f"i want to take DA125 instead on 09/01, please help me modify my itinerary {confirmation_number}")
print(result)
```

confirmation_number = "{copy the confirmation number here}" result = agent(user_request=f"i want to take DA125 instead on 09/01, please help me modify my itinerary {confirmation_number}") print(result)

```
Prediction(
    trajectory={'thought_0': "I need to fetch the user's current itinerary using the confirmation number provided (7zokt5v5) to understand the details of their existing booking before making any modifications.", 'tool_name_0': 'fetch_itinerary', 'tool_args_0': {'confirmation_number': '7zokt5v5'}, 'observation_0': None, 'thought_1': "I need to fetch the user's current itinerary using the confirmation number provided (7zokt5v5) to understand the details of their existing booking before making any modifications.", 'tool_name_1': 'fetch_itinerary', 'tool_args_1': {'confirmation_number': '7zokt5v5'}, 'observation_1': None, 'thought_2': "I need to file a customer support ticket since I am unable to fetch the user's itinerary with the provided confirmation number.", 'tool_name_2': 'file_ticket', 'tool_args_2': {'user_request': 'i want to take DA125 instead on 09/01, please help me modify my itinerary 7zokt5v5', 'user_profile': {'user_id': '', 'name': '', 'email': ''}}, 'observation_2': 'lf3n2t', 'thought_3': "Since I have filed a customer support ticket for the user's request, I will wait for a response from the support team regarding the modification of the itinerary.", 'tool_name_3': 'finish', 'tool_args_3': {}, 'observation_3': 'Completed.'},
    reasoning="I attempted to modify the user's itinerary by fetching the current details using the provided confirmation number (7zokt5v5). However, I was unable to retrieve the itinerary. As a result, I filed a customer support ticket to address the issue and request the modification to the user's itinerary for flight DA125 on 09/01.",
    process_result='A customer support ticket has been filed regarding your request to modify your itinerary. The ticket number is lf3n2t. Please wait for a response from the support team for further assistance.'
)
```

## Conclusion[¶](#conclusion)

Congrats on finishing the tutorial! In this tutorial we have seen how to build a customer service agent with DSPy. The gists are:

- Define the tools as python function, and **add docstring and type hints**.
- Provide the tools to `dspy.ReAct` along with a signature to define the task.
- Invoke the `dspy.ReAct` with the inputs field defined in the signature, and it will start the reasoning and acting loop behind the scene.
