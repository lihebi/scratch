#lang turnstile/quicklang

(provide ann λ def
         #%datum
         #%app
         ; if let rec #%app #%datum
         (rename-out [λ lambda]))
(provide (type-out Bool Int ->))
(provide (typed-out [not    (-> Bool Bool)]
                    [+      (-> Int Int Int)]
                    [*      (-> Int Int Int)]
                    [/      (-> Int Int Int)]
                    [<=     (-> Int Int Bool)]
                    [zero?  (-> Int Bool)]))

(require rackunit/turnstile)
(provide (all-from-out rackunit/turnstile))
(require turnstile/no-unicode)


;; Types

(define-base-types Bool Int)
(define-type-constructor -> #:arity >= 1)


;; Expression Syntax

; (ann :expr :type)
;
; Type annotation. Infers the given type if the expression can be
; checked to have that type.
(define-typed-syntax ann
  ; to COMPUTE the type of ‘(ann e τ)’, . . .
  [(_ e:expr τ:type)    
   ≫
   ; CHECK that ‘e’ has type ‘τ’ (normalized), calling its
   ; expansion ‘e-’, . . .
   [⊢ e ≫ e- ⇐ τ.norm] 
   ----
   ; then expand to ‘e-’ and COMPUTE type ‘τ’ (normalized).
   [⊢ e- ⇒ τ.norm]])


; (λ (:id ...) :expr)
; (λ ([:id :type] :expr)
;
; Function abstraction. The first form has the same syntax as
; Racket, but is checking-mode only.
(define-typed-syntax λ
  [(_ (x:id ...) e:expr)
   ⇐ (~-> s ... t)
   ≫
   #:fail-when (check-duplicate-identifier (stx->list #'(x ...)))
   "repeated formal parameter name"
   #:fail-unless (= (stx-length #'(x ...)) (stx-length #'(s ...)))
   "wrong number of formal parameters for expected arrow type"
   [[x ≫ x- : s] ... ⊢ e ≫ e- ⇐ t]
   ----
   [⊢ (λ- (x- ...) e-)]]
  
  [(_ ([x:id σ:type] ...) e:expr)
   ≫
   #:fail-when (check-duplicate-identifier (stx->list #'(x ...)))
   "repeated formal parameter name"
   #:with (s ...) #'(σ.norm ...)
   [[x ≫ x- : s] ... ⊢ e ≫ e- ⇒ t]
   ----
   [⊢ (λ- (x- ...) e-) ⇒ (-> s ... t)]])


;;
;; Definition Syntax
;;

; (def x:id e:expr)
; (def x:id t:type e:expr)

;
; Module-level variable definition. Unlike Racket’s define, def doesn't
; do recursion; its scope is downward-only.
(define-typed-syntax def
  [(_ x:id e:expr)
   ≫
   ----
   [≻ (define-typed-variable x e)]]

  [(_ x:id τ:type e:expr)
   ≫
   [⊢ e ≫ e- ⇐ τ.norm]
   ----
   [≻ (define-typed-variable x e- ⇐ τ.norm)]])


(define-typed-syntax #%datum
  [(_ . x:integer)
   ≫
   ;; [/- x >> (#%datum- x-) <= Int]
   ----
   [/- (#%datum- . x) => Int]]
  [(_ . x:boolean)
   ≫
   ;; [/- x >> (#%datum- x-) <= Bool]
   ----
   [/- (#%datum- . x) => Bool]])

(define-typed-syntax #%app
  [(_ f e ...)
   ;; FIXME arity check
   >>
   [/- f >> f- => (~-> s ... t)]
   [/- e >> e- <= s] ...
   ----
   [/- (#%app- f- e- ...) => t]])

#;
(define-typed-syntax #%app
  [(_ e_fn e_arg ...) ≫
   [⊢ e_fn ≫ e_fn- ⇒ (~-> τ_in ... τ_out)]
   [⊢ e_arg ≫ e_arg- ⇐ τ_in] ...
   --------
   [⊢ (#%app- e_fn- e_arg- ...) ⇒ τ_out]])


(define-typed-syntax if
  [(_ c t f)
   >>
   [/- c >> c- => Bool]
   [/- t >> t- => τ]
   [/- f >> f- => σ]
   ----
   [/- (if- c- t- f-) =>
       ;; FIXME when type=? fail, what to return?
       #,(when (type=? #'τ #'σ)
           #'τ)]])

