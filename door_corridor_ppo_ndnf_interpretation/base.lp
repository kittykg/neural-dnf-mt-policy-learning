% The base idea of the interpretation is map a observation either positively or
% negatively to an image encoding.
% The mapping between observations and image encodings should be n to 1, i.e.,
% many observations can be needed by an image encoding.
% But an observation with a sign can't be mapped to more than one encoding.
% The mapping is represented as include(a(I), obs(X, Y, O, S), Sign)


% TODO:
% At each time, check the action
% Ensure the image encoding(s) needed is fired for that action
% So need to change the definition of `fired_img_encoding`

% ------------------------------------------------------------------------------
% ------------------------------------------------------------------------------

% fired_img_encoding/2
% This predicate is true if under the obs-enc mapping, image encoding a(I) is
% fired at time T. It requires all the observations needed by the image encoding
% to be fired correctly with the correct sign.
fired_img_encoding_from_inclusion(Time, a(I)) :-
    timestamp(Time),
    img_encoding_id(I),
    #count { X, Y, O, S: include_observation_for_encoding(Time, obs(X, Y, O, S), a(I))} = Count,
    #count { X, Y, O, S: include(a(I), obs(X, Y, O, S), Sign)} = Count,
    Count > 0.

% include_observation_for_encoding/3
% This predicate is the helper predicate for fired_img_encoding/2. It is true if
% the mapping between the observation and the encoding is positive; or the
% mapping between the observation and the encoding is negative.
include_observation_for_encoding(Time, obs(X, Y, O, S), a(I)) :-
    timestamp(Time),
    img_encoding_id(I),
    is_possible_observation(obs(X, Y, O, S)),
    include(a(I), obs(X, Y, O, S), pos),
    fired_observation(Time, obs(X, Y, O, S)).

include_observation_for_encoding(Time, obs(X, Y, O, S), a(I)) :-
    timestamp(Time),
    img_encoding_id(I),
    is_possible_observation(obs(X, Y, O, S)),
    include(a(I), obs(X, Y, O, S), neg),
    not fired_observation(Time, obs(X, Y, O, S)).

% ------------------------------------------------------------------------------
%                               Hard constraint
% ------------------------------------------------------------------------------

% For a given observation, it can't be both positively/negatively mapped to
% two different image encodings.
% :- img_encoding_id(I), img_encoding_id(J), is_possible_observation(obs(X, Y, O, S)),
%     include(a(I), obs(X, Y, O, S), Sign), include(a(J), obs(X, Y, O, S), Sign), I != J.

% An image encoding has to have at least one observation needed.
:- img_encoding_id(I), #count{ X, Y, O, S: include(a(I), obs(X, Y, O, S), Sign) } = 0.

% If an image encoding a(I) is required to fire at time T, but with the mapping
% it is not fired, then it is a violation.
:- timestamp(T), img_encoding_id(I), required_img_encoding(T, a(I), pos),
    not fired_img_encoding_from_inclusion(T, a(I)).

% If an image encoding a(I) is required to not fire at time T, but with the
% mapping it is fired, then it is a violation.
:- timestamp(T), img_encoding_id(I), not required_img_encoding(T, a(I), pos),
    fired_img_encoding_from_inclusion(T, a(I)).

% ------------------------------------------------------------------------------
%                               Soft constraint
% ------------------------------------------------------------------------------

% Soft constraints (optimisation):
% Reduce the number of mapping needed for an image encoding. Penalise at level 1
% for penalty 1.
:~ img_encoding_id(I), is_possible_observation(obs(X, Y, O, S)),
    include(a(I), obs(X, Y, O, S), _). [2@1]

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

% #show fired_img_encoding/2.
% #show fired_img_encoding_from_inclusion/2.
% #show include_observation_for_encoding/3.
#show include/3.
% #show counting/2.
