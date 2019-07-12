#lang racket
(require "run.rkt"
         (for-syntax syntax/parse
                     racket/list)
         syntax/wrap-modbeg)
 
(provide
 ;; #%module-begin
 ;; my-module-begin
 #%top-interaction
 &&
 (rename-out [pfsh:run run]
             [mydefine define]
             [my-module-begin #%module-begin]
             ;; [#%module-begin #%module-begin]
             ))

#;
(define-syntax (my-wrapper stx)
  (displayln stx)
  #`(void #,(rest stx)))
(define-syntax (my-wrapper stx)
  (syntax-parse stx
    [(_ e)
     #'(void e)]))

(define-syntax my-module-begin
  (make-wrapping-module-begin
   ;; #'my-wrapper
   #'void
   ))

(define-syntax (pfsh:run stx)
  (syntax-parse stx
    [(_ prog:id arg:id ... (~datum <) input)
     #`(with-input-from-string input
         (λ () (run
                (as-string prog)
                (as-string arg) ...)))]
    [(_ prog:id arg:id ...)
     #`(run (as-string prog) (as-string arg) ...)]
    [(_ prog:id arg:string ...)
     #`(run (as-string prog) 'arg ...)]
    ))
 
(define-syntax (as-string stx)
  (syntax-parse stx
    [(_ sym:id)
     #`#,(symbol->string (syntax-e #'sym))]))

(define-syntax (mydefine stx)
  (syntax-parse stx
    [(_ name:id e:expr)
     #'(define name (with-output-to-string (λ () e)))]))

(define-syntax (&& stx)
  (syntax-parse stx
    [(_ e)
     #'e]
    [(_ e other ...)
     #'(when e (&& other ...))]))
