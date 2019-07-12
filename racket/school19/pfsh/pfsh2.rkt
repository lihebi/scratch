#lang racket
(require "run.rkt"
         (for-syntax syntax/parse)
         syntax/wrap-modbeg)
 
(provide
 #%module-begin
 ;; (make-wrapping-module-begin)
 #%top-interaction
 ;; define
 ;; string-append
 (rename-out [pfsh:run run]
             [mydefine define]))

;; TODO
#;
(begin-for-syntax
  (define-syntax-class (run-arg)
    (pattern [x:id])
    (pattern [x:string])))

(define-syntax (pfsh:run stx)
  (syntax-parse stx
    [(_ prog:id arg:id ... (~datum <) input)
     #`(void (with-input-from-string input
               (λ () (run
                      (as-string prog)
                      (as-string arg) ...))))]
    [(_ prog:id arg:id ...)
     ;; #:declare arg (or identifier char)
     #`(void (run (as-string prog) (as-string arg) ...))]
    [(_ prog:id arg:string ...)
     #`(void (run (as-string prog) 'arg ...))]
    ))
 
(define-syntax (as-string stx)
  (syntax-parse stx
    [(_ sym:id)
     #`#,(symbol->string (syntax-e #'sym))]))

(define-syntax (mydefine stx)
  (syntax-parse stx
    [(_ name:id e:expr)
     #'(define name (with-output-to-string (λ () e)))]))
