%用普通 SWI-Prolog 写的一段元解释器（meta-interpreter）+ 迭代加深搜索（IDS）+ 证明树生成的代码。
%两套“执行器”（meta-interpreter 内核）
%A. 带证明树（proof tree generation）
%不限制深度：mi_tree/2
%mi_tree(Goal, Proof)：证明 Goal，并构造 Proof。
%限制深度：mi_limit/4
%mi_limit(Goal, Proof, DepthIn, DepthOut)：在深度预算内证明，并构造 Proof。
%B. 不生成证明树（no_proof）
%限制深度：mi_limit_no_proof/3（以及它的壳 mi_limit_no_proof/2）

%mi_limit_no_proof(Goal, DepthIn, DepthOut)：只判断能不能证明成功，不构造 Proof。

% Prolog

%加载库
%请加载 clpfd 这个库，让我能用 #=、#> 等整数约束功能
:- use_module(library(clpfd)). #library(clpfd)：约束逻辑整数库（Finite Domain）

% set_prolog_flag(answer_write_options, [max_depth(30)]).


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                  To write all answers in one query          %
%                  So that we do not need to hit ";".         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Output all answers at once, without hitting ";"
writeall(Q) :- forall(Q,writeln(Q)).


% to limit the number of solutions to be written out
% more advanced than "writeall\1"

% Main predicate
write_max_solutions(Q, Max) :-
    write_solutions_up_to(Q, Max, 1).

% Recursive predicate to stop after Max solutions
write_solutions_up_to(_, Max, Counter) :-
    Counter > Max, 
    !.  % Cut to ensure it doesn't backtrack into infinite solutions

write_solutions_up_to(Q, Max, Counter) :-
    once(Q),  % Retrieves only one solution without backtracking for more
    writeln(Q),
    NewCounter is Counter + 1,
    write_solutions_up_to(Q, Max, NewCounter).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                      meta-interpreters                      %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

g(G) :- call(G).

% Clause lookup with defaulty betterment
mi_clause(G, G) :-
    %如果 G 是内建谓词（built-in），就直接返回自己
    predicate_property(G, built_in), !.
mi_clause(G, Body) :-
    %否则：从数据库里取出 G 的定义（body），再做改写
    clause(G, B),
    defaulty_better(B, Body).


defaulty_better(true, true).
defaulty_better((A,B), (BA,BB)) :-
        defaulty_better(A, BA),
        defaulty_better(B, BB).
defaulty_better(G, g(G)) :-
        G \= true,
        G \= (_,_).

% Define the operator for proofs
%把 => 这个符号定义成一个“可以当中缀运算符用的符号”，用来写证明树（proof）。
:- op(750, xfy, =>).

%生成证明树（不限制深度）
% Proof tree generation
%true 在 Prolog 里表示“永远成功 / 什么都不做”
mi_tree(true, true).
        mi_tree(A, TA),
        mi_tree(B, TB).
%如果当前要证明的目标 G 是 Prolog 自带的“内建谓词”(built-in)，那就直接执行它，并在证明树里把它记成一个叶子节点 builtin(G)
mi_tree(G, builtin(G)) :- predicate_property(G, built_in), !, call(G).
mi_tree(g(G), TBody => G) :-
        mi_clause(G, Body),
        mi_tree(Body, TBody).

%带深度限制的证明树生成
% Depth-limited meta-interpreter with proof tree generation
mi_limit(true, true, N, N).
mi_limit((A,B), (TA,TB), N0, N) :-
        mi_limit(A, TA, N0, N1),
        mi_limit(B, TB, N1, N).
mi_limit(G, builtin(G), N, N) :- predicate_property(G, built_in), !, call(G). % **This line seems to make iterative-deepening search not work.**
mi_limit(g(G), TBody => G, N0, N) :-
        N0 #> 0,
        N1 #= N0 - 1,
        mi_clause(G, Body),
        mi_limit(Body, TBody, N1, N).

%迭代加深搜索（Iterative Deepening Search, IDS）
%无限迭代加深（N = 0,1,2,3,…）
% Iterative deepening with proof tree generation
mi_id(Goal, Proof) :-
        length(_, N),
        mi_limit(Goal, Proof, N, _).

%有上限的迭代加深（N = 1..MaxDepth）
% Iterative deepening with maximum depth with proof tree generation
mi_id_limit(Goal, Proof, MaxDepth) :-
        between(1, MaxDepth, N),
        mi_limit(Goal, Proof, N, _).

% Sample Usage
% mi_id(some_goal, Proof)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% mi_clause(G, Body) :-
%         clause(G, B),
%         defaulty_better(B, Body).

% defaulty_better(true, true).
% defaulty_better((A,B), (BA,BB)) :-
%         defaulty_better(A, BA),
%         defaulty_better(B, BB).
% defaulty_better(G, g(G)) :-
%         G \= true,
%         G \= (_,_).


% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% :- op(750, xfy, =>).

% mi_tree(true, true).
% mi_tree((A,B), (TA,TB)) :-
%         mi_tree(A, TA),
%         mi_tree(B, TB).
% mi_tree(g(G), TBody => G) :-
%         mi_clause(G, Body),
%         mi_tree(Body, TBody).


% Below Code limits the depth of the search tree. From https://www.metalevel.at/acomip/
mi_limit_no_proof(Goal, Max) :-
        mi_limit_no_proof(Goal, Max, _).

mi_limit_no_proof(true, N, N).
mi_limit_no_proof((A,B), N0, N) :-
        mi_limit_no_proof(A, N0, N1),
        mi_limit_no_proof(B, N1, N).
mi_limit_no_proof(g(G), N0, N) :-
        N0 #> 0,
        N1 #= N0 - 1,
        mi_clause(G, Body),
        mi_limit_no_proof(Body, N1, N).

% Below is iterative deepening search, no proof tree generation
mi_id_limit_no_proof(Goal, MaxDepth) :-
        between(1, MaxDepth, N),
        mi_limit_no_proof(Goal, N).

% How to use this?
% mi_tree(g(Goal), T)
% mi_limit_no_proof(g(Goal), 3).
% mi_id(g(Goal)).