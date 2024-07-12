% Defining what observations has to hold at a given time for a predicate to be
% true at that same given time. Atom `a(I)` refers to policy atom with id `I`
% where `I` is an integer
% e.g.
% include(a(5), obs(1, 1, 3, 1), pos).
% include(a(5), obs(2, 2, 3, 1), neg).

-fired_img_encoding_gt(T, a(I)) :-
    timestamp(T), img_encoding_id(I),
    not fired_img_encoding_gt(T, a(I)).

% What facts are true at some given time: `fired_obs_at_timestep(T, Obs)`
% Generate classic negation from NAF
% e.g.
% fired_obs_at_timestep(1, obs(1, 1, 3, 1)).
% -fired_obs_at_timestep(1, obs(2, 2, 3, 1)).
-fired_obs_at_timestep(T, obs(X, Y, O, S)) :-
    timestamp(T), is_possible_observation(obs(X, Y, O, S)),
    not fired_obs_at_timestep(T, obs(X, Y, O, S)).

% Determine if a policy atom holds for a given time step
-a_fired_at_timestep(T, a(I)) :-
    timestamp(T), img_encoding_id(I),
    is_possible_observation(obs(X, Y, O, S)),
    include(a(I), obs(X, Y, O, S), pos),
    -fired_obs_at_timestep(T, obs(X, Y, O, S)).

-a_fired_at_timestep(T, a(I)) :-
    timestamp(T), img_encoding_id(I),
    is_possible_observation(obs(X, Y, O, S)),
    include(a(I), obs(X, Y, O, S), neg),
    fired_obs_at_timestep(T, obs(X, Y, O, S)).

a_fired_at_timestep(T, a(I)) :-
    timestamp(T), img_encoding_id(I),
    not -a_fired_at_timestep(T, a(I)).

% -a_fired_at_timestep(T, a(I)) :-
%     timestamp(T), img_encoding_id(I),
%     not a_fired_at_timestep(T, a(I)).

% ------------------------------------------------------------------------------
%                               Hard constraint
% ------------------------------------------------------------------------------

% An image encoding has to have at least one observation needed.
:- img_encoding_id(I), #count{ X, Y, O, S: include(a(I), obs(X, Y, O, S), Sign) } = 0.

% An observation with `Sign` can't bet used by two different image encodings
:- img_encoding_id(I), img_encoding_id(J), is_possible_observation(obs(X, Y, O, S)),
    include(a(I), obs(X, Y, O, S), Sign), include(a(J), obs(X, Y, O, S), Sign), I != J.

% If an image encoding is fired at time T, then using the inclusion mapping, the
% image encoding should be fired to match it
:- timestamp(T), img_encoding_id(I), fired_img_encoding_gt(T, a(I)),
    -a_fired_at_timestep(T, a(I)).

% If an image encoding is not fired at time T, then using the inclusion mapping,
% the image encoding should not be fired to match it
:- timestamp(T), img_encoding_id(I), -fired_img_encoding_gt(T, a(I)),
    a_fired_at_timestep(T, a(I)).

% Using the mapping, at each time step, only one action (`disj`) can be fired
% (this is the mutual exclusivity constraint on action)
:- timestamp(T), disj(T, A), disj(T, B), disj_id(A), disj_id(B), A < B.

% Never include the agent location observation for any image encoding
:- img_encoding_id(I), is_possible_observation(obs(X, Y, 4, S)),
    include(a(I), obs(X, Y, 4, S), _).

% ------------------------------------------------------------------------------
%                               Soft constraint
% ------------------------------------------------------------------------------

% Soft constraints (optimisation):
% Reduce the number of mapping needed for an image encoding
#minimize { Count @ 3, I : img_encoding_id(I), counting(a(I), Count) }.

% Avoid using the same observation with different sign for two different image
% encodings
:~ img_encoding_id(I), img_encoding_id(J), is_possible_observation(obs(X, Y, O, S)),
    include(a(I), obs(X, Y, O, S), pos), include(a(J), obs(X, Y, O, S), neg), I != J. [1@2]

% Avoid using observation with sign `neg` for an image encoding
#minimize { 1 @ 1, I: include(a(I), obs(X, Y, O, S), neg) }.

% ------------------------------------------------------------------------------
%                               Helper predicates
% ------------------------------------------------------------------------------

% counting/2
% Count how many observations are needed for an image encoding
counting(a(I), Count) :-
    img_encoding_id(I),
    #count { X, Y, O, S: include(a(I), obs(X, Y, O, S), Sign)} = Count.

% ------------------------------------------------------------------------------
%                               Show statements
% ------------------------------------------------------------------------------

% #show fired_img_encoding_gt/2.
% #show -fired_img_encoding_gt/2.
% #show a_fired_at_timestep/2.
% #show -a_fired_at_timestep/2.
% #show disj/2.
% #show counting/2.
#show include/3.
