{-# LANGUAGE ExistentialQuantification #-}
module Main where
-- for getArgs
import System.Environment
-- for liftM
import Control.Monad
-- for error handling
import Control.Monad.Error
-- import Control.Monad.Trans.Error

import Text.ParserCombinators.Parsec hiding (spaces)
import System.Environment

import System.IO

main :: IO ()
main = do
  args <- getArgs
  case length args of
    0 -> runRepl
    1 -> evalAndPrint $ args !! 0
    otherwise -> putStrLn "Program takes 0 or 1 argument"

flushStr :: String -> IO ()
flushStr str = putStr str >> hFlush stdout

readPrompt :: String -> IO String
readPrompt prompt = flushStr prompt >> getLine
evalString :: String -> IO String
evalString expr = return $ extractValue $ trapError (liftM show $ readExpr expr >>= eval)
evalAndPrint :: String -> IO ()
evalAndPrint expr = evalString expr >>= putStrLn

-- naming convention: ends with an underscore: monadic functions that
-- repeat but do not return a value
--
-- using Monad m => and m a because, the corresponding parameter is
-- generalized on all monads, not just IO
until_ :: Monad m => (a -> Bool) -> m a -> (a -> m ()) -> m()
-- 1. pred: signal when to stop
-- 2. prompt: m a, an action to perform before the test (to get result in this case)
-- 3. action: a function returning an action, to deal with the input
until_ pred prompt action = do
  result <- prompt
  if pred result
     then return ()
     else action result >> until_ pred prompt action

runRepl :: IO ()
runRepl = until_ (== "quit") (readPrompt "Lisp>>> ") evalAndPrint

myadd :: Num a => a -> a -> a
myadd a b = a + b

symbol :: Parser Char
symbol = oneOf "!#$%&|*+-/:<=>?@^_~"

readExpr :: String -> ThrowsError LispVal
readExpr input = case parse
  -- (spaces >> symbol)
  parseExpr
  "lisp" input of
                   -- the parse function returns an Either, it
                   -- contains two data constructors, Left for error,
                   -- Right for success and gives value
                   -- throwError is built-in
                   Left err -> throwError $ Parser err
                   Right val -> return val

spaces :: Parser ()
spaces = skipMany1 space

data LispVal = Atom String
             | List [LispVal]
             | DottedList [LispVal] LispVal
             | Number Integer
             | String String
             | Bool Bool

-- used return to lift a String (one of the LispVal constructor) into
-- the Parser monad
parseString :: Parser LispVal
parseString = do
  -- _ <- is used to suppress the warning of discard a value of do
  -- notation
  _ <- char '"'
  -- rules to use >>, >>=, do
  -- if no passing values, use >>
  -- if parsing value to next expression, use >>=
  -- otherwise use "do" to bind
  x <- many (noneOf "\"")
  _ <- char '"'
  -- $ is a infix function applicatoin
  -- it is equivalent to return (String x)
  -- it has right associcativity, low precedence
  return $ String x

parseAtom :: Parser LispVal
parseAtom = do
  -- <|> is in Parsec, a combinator semantically, it tries one by one,
  -- until one succeeds. But there's one requirement: it must be LL(1),
  -- i.e. it cannot consume. Parsec allows backtracking, via "try"
  first <- letter <|> symbol
  rest <- many (letter <|> digit <|> symbol)
  -- Connect first and rest into a list, using cons operator (:).
  -- This is equivalent to using [first] ++ rest.  Note that first is
  -- not a list, thus it must be bracketed.
  let atom = first:rest
  return $ case atom of
    "#t" -> Bool True
    "#f" -> Bool False
    _ -> Atom atom

parseNumber :: Parser LispVal
-- many1 is a Parsec combinator that matches one or more, it returns a
-- string. We first use read to convert it into a number, then use
-- Number constructor to construct a LispVal.
--
-- the . here is a function composition operator. It creates a
-- function that first apply right function, then apply left function.
--
-- However, many1 will not return a raw string, but a Parser
-- String. So applying liftM on the function (Number . read) will let
-- it able to consume the String inside the monad.
-- liftM is inside Control.Monad
parseNumber = liftM (Number . read) $ many1 digit

-- using liftM is not that beautiful, so consider these alternatives
-- alternative 1
parseNumber1 :: Parser LispVal
parseNumber1 = do x <- many1 digit
                  (return . Number . read) x
-- alternative 2
parseNumber2 :: Parser LispVal
parseNumber2 = many1 digit >>= \x -> (return . Number . read) x


parseExpr :: Parser LispVal
parseExpr = parseAtom
            <|> parseString
            <|> parseNumber
            <|> parseQuoted
            <|> do char '('
                   -- try here is to allow backtracking.  It tries one
                   -- by one, and if it fails, it backs up the
                   -- previous state
                   x <- try parseList <|> parseDottedList
                   char ')'
                   return x

parseList :: Parser LispVal
parseList = liftM List $ sepBy parseExpr spaces

parseDottedList :: Parser LispVal
parseDottedList = do
  head1 <- endBy parseExpr spaces
  tail1 <- char '.' >> spaces >> parseExpr
  return $ DottedList head1 tail1

parseQuoted :: Parser LispVal
parseQuoted = do
  char '\''
  x <- parseExpr
  return $ List [Atom "quote", x]




-- evaluation

showVal :: LispVal -> String
showVal (String contents) = "\"" ++ contents ++ "\""
showVal (Atom name) = name
showVal (Number contents) = show contents
showVal (Bool True) = "#t"
showVal (Bool False) = "#f"
showVal (List contents) = "(" ++ unwordsList contents ++ ")"
showVal (DottedList head tail) = "(" ++ unwordsList head
                                 ++ " . " ++ showVal tail
                                 ++ ")"

unwordsList :: [LispVal] -> String
-- point-free style: writing definitions purely in terms of function
-- composition and partial application
-- unwords is a Haskell function that add spaces between list items
unwordsList = unwords . map showVal

-- support show on LispVal
instance Show LispVal where show = showVal


eval :: LispVal -> ThrowsError LispVal
eval val@(String _) = return val
eval val@(Number _) = return val
eval val@(Bool _) = return val
eval (List [Atom "quote", val]) = return val

eval (List [Atom "if", pred, conseq, alt]) =
  do result <- eval pred
     case result of
       Bool False -> eval alt
       otherwise -> eval conseq

-- mapM maps a monadic function over a list
-- it returns Right [results] on success, Left error on failure
eval (List (Atom func : args)) = mapM eval args >>= apply func

eval badForm = throwError $ BadSpecialForm "Unrecognized special form" badForm

apply :: String -> [LispVal] -> ThrowsError LispVal
-- lookup is a Haskell function that looks up a key inside a list of
-- pairs. It returns a Maybe.
apply func args = maybe (throwError $ NotFunction "Unrecognized primitive function args" func)
                        ($ args)
                        (lookup func primitives)

primitives :: [(String, [LispVal] -> ThrowsError LispVal)]
primitives = [("+", numericBinop (+)),
              ("-", numericBinop (-)),
              ("*", numericBinop (*)),
              ("/", numericBinop div),
              ("mod", numericBinop mod),
              ("quotient", numericBinop quot),
              ("remainder", numericBinop rem),
              ("=", numBoolBinop (==)),
              ("<", numBoolBinop (<)),
              (">", numBoolBinop (>)),
              ("/=", numBoolBinop (/=)),
              (">=", numBoolBinop (>=)),
              ("<=", numBoolBinop (<=)),
              ("&&", boolBoolBinop (&&)),
              ("||", boolBoolBinop (||)),
              ("string=?", strBoolBinop (==)),
              ("string<?", strBoolBinop (<)),
              ("string>?", strBoolBinop (>)),
              ("string<=?", strBoolBinop (<=)),
              ("string>=?", strBoolBinop (>=)),
              ("car", car),
              ("cdr", cdr),
              ("cons", cons),
              ("eq?", eqv),
              ("eqv?", eqv),
              ("equal?", equal)]

-- takes a primitive haskell function
-- wrap it with code to unpack an argument list
-- apply the function to the arguments
-- wrap result in Number constructor
--
-- But why the op has (Int -> Int -> Int)? This can support arbitrary
-- operants??
numericBinop :: (Integer -> Integer -> Integer) -> [LispVal] -> ThrowsError LispVal

numericBinop op [] = throwError $ NumArgs 2 []
numericBinop op singleVal@[_] = throwError $ NumArgs 2 singleVal
-- foldl1 is said to implement arbitrary operants support
numericBinop op params = mapM unpackNum params >>= return . Number . foldl1 op

unpackNum :: LispVal -> ThrowsError Integer
unpackNum (Number n) = return n
-- reads is a Haskell function that parse a string into a pair (parsed
-- number, rest string)
unpackNum (String n) = let parsed = reads n :: [(Integer, String)] in
  if null parsed
     then throwError $ TypeMismatch "number" $ String n
     -- !! means index 0. Then why need fst??
     else return $ fst $ parsed !! 0
-- match one-item list
unpackNum (List [n]) = unpackNum n
-- otherwise
unpackNum notNum = throwError $ TypeMismatch "number" notNum

boolBinop :: (LispVal -> ThrowsError a) -> (a -> a -> Bool) -> [LispVal] -> ThrowsError LispVal
boolBinop unpacker op args = if length args /= 2
  then throwError $ NumArgs 2 args
  else do left <- unpacker $ args !! 0
          right <- unpacker $ args !! 1
          -- any function can be turned into an infix operator by
          -- wrapping it in backticks
          return $ Bool $ left `op` right

numBoolBinop = boolBinop unpackNum
strBoolBinop = boolBinop unpackStr
boolBoolBinop = boolBinop unpackBool

unpackStr :: LispVal -> ThrowsError String
unpackStr (String s) = return s
unpackStr (Number s) = return $ show s
unpackStr (Bool s) = return $ show s
unpackStr notString = throwError $ TypeMismatch "string" notString

unpackBool :: LispVal -> ThrowsError Bool
unpackBool (Bool b) = return b
unpackBool notBool = throwError $ TypeMismatch "boolean" notBool


car :: [LispVal] -> ThrowsError LispVal
-- using [] outside because we want [LispVal], instead of one LispVal
-- that is a list
car [List (x : xs)] = return x
car [DottedList (x : xs) _] = return x
car [badArg] = throwError $ TypeMismatch "pair" badArg
car badArgList = throwError $ NumArgs 1 badArgList

cdr :: [LispVal] -> ThrowsError LispVal
cdr [List (x : xs)] = return $ List xs
cdr [DottedList [_] x] = return x
cdr [DottedList (_ : xs) x] = return $ DottedList xs x
cdr [badArg] = throwError $ TypeMismatch "pair" badArg
cdr badArgList = throwError $ NumArgs 1 badArgList

cons :: [LispVal] -> ThrowsError LispVal
cons [x1, List []] = return $ List [x1]
cons [x, List xs] = return $ List $ x : xs
cons [x, DottedList xs xlast] = return $ DottedList (x : xs) xlast
cons [x1, x2] = return $ DottedList [x1] x2
cons badArgList = throwError $ NumArgs 2 badArgList

eqv :: [LispVal] -> ThrowsError LispVal
eqv [(Bool arg1), (Bool arg2)] = return $ Bool $ arg1 == arg2
eqv [(Number arg1), (Number arg2)] = return $ Bool $ arg1 == arg2
eqv [(String arg1), (String arg2)] = return $ Bool $ arg1 == arg2
eqv [(Atom arg1), (Atom arg2)] = return $ Bool $ arg1 == arg2
eqv [(DottedList xs x), (DottedList ys y)] = eqv [List $ xs ++ [x], List $ ys ++ [y]]
eqv [(List arg1), (List arg2)]
  = return $ Bool $ (length arg1 == length arg2) &&
    -- zip seems to pair each corresponding items in the two list
    -- all will apply the function, and succeeds if all of them succeed
    (all eqvPair $ zip arg1 arg2)
  -- this is a local function definition
  where eqvPair (x1, x2) = case eqv [x1, x2] of
          Left err -> False
          Right (Bool val) -> val
eqv [_,_] = return $ Bool False
eqv badArgList = throwError $ NumArgs 2 badArgList

-- need to use {-# LANGUAGE ExistentialQuantification #-}
data Unpacker = forall a. Eq a => AnyUnpacker (LispVal -> ThrowsError a)
unpackEquals :: LispVal -> LispVal -> Unpacker -> ThrowsError Bool
unpackEquals arg1 arg2 (AnyUnpacker unpacker) =
  do unpacked1 <- unpacker arg1
     unpacked2 <- unpacker arg2
     return $ unpacked1 == unpacked2
  `catchError` (const $ return False)

equal :: [LispVal] -> ThrowsError LispVal
equal [arg1, arg2] = do
  -- map (try) all unpackers, and if any of them equal, we treat it as equal
  primitiveEquals <- liftM or $ mapM (unpackEquals arg1 arg2)
                     -- this is a heterogenous list
                     [AnyUnpacker unpackNum, AnyUnpacker unpackStr, AnyUnpacker unpackBool]
  eqvEquals <- eqv [arg1, arg2]
  return $ Bool $ (primitiveEquals || let (Bool x) = eqvEquals in x)

-- Error handling

data LispError = NumArgs Integer [LispVal]
               | TypeMismatch String LispVal
               | Parser ParseError
               | BadSpecialForm String LispVal
               | NotFunction String String
               | UnboundVar String String
               | Default String
showError :: LispError -> String
showError (UnboundVar message varname)
  = message ++ ": " ++ varname
showError (BadSpecialForm message form)
  = message ++ ": " ++ show form
showError (NotFunction message func)
  = message ++ ": " ++ show func
showError (NumArgs expected found)
  = "Expected " ++ show expected
  ++ " args; found values " ++ unwordsList found
showError (TypeMismatch expected found)
  = "Invalid type: expected " ++ expected
  ++ ", found " ++ show found
showError (Parser parseErr)
  = "Parse error at " ++ show parseErr

instance Show LispError where show = showError

-- Make it an instance of the builtin Error class.  This is necessary
-- for GHC's built-in error handling functions.
instance Error LispError where
  noMsg = Default "An error has occurred"
  strMsg = Default

-- This is partial evaluation. The rule for type acts same as
-- functions.
type ThrowsError = Either LispError

-- Either has two other functions apart from monadic ones
-- 1. throwError: takes an Error value, and lifts it into Left constructor
-- 2. catchError: takes an action and return a normal value ???
trapError action = catchError action (return . show)
-- the result of trapError will be an Either always with Right value,
-- i.e. it is valid ???

-- extract the value from our ThrowsError (which is an Either monad)
-- notice that we left the Left case, because we want it to just fail
extractValue :: ThrowsError a -> a
extractValue (Right val) = val

