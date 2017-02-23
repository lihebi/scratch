#include "clang/AST/ASTConsumer.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendAction.h"
#include "clang/Tooling/Tooling.h"

#include "clang/Frontend/FrontendActions.h"

#include "llvm/Support/CommandLine.h"
#include "clang/Tooling/CommonOptionsParser.h"

using namespace clang;

using namespace clang::tooling;
using namespace llvm;

static cl::OptionCategory MyToolCategory("my-tool options");


class FunctionVisitor
  : public RecursiveASTVisitor<FunctionVisitor> {
public:
  explicit FunctionVisitor(ASTContext *Context)
    : Context(Context) {}
  bool VisitStmt(Stmt *S) {
    S->dump();
    return true;
  }
private:
  ASTContext *Context;
};


/**
 * 1. Get the translation unit
 * 2. Get all functions
 * 3. Dump the AST of the function
 * 4. Break into tokens
 */


// I need to define a bunch of VisitXXX function for each type of node
class TokenVisitor
  : public RecursiveASTVisitor<TokenVisitor> {
public:
  // receiving the context for accessing loc
  explicit TokenVisitor(ASTContext *Context)
    : Context(Context) {
    const SourceManager &manager = Context->getSourceManager();
    m_main_entry = manager.getFileEntryForID(manager.getMainFileID());
  }

  bool VisitCXXRecordDecl(CXXRecordDecl *Declaration) {
    if (Declaration->getQualifiedNameAsString() == "n::m::C") {
      FullSourceLoc FullLocation = Context->getFullLoc(Declaration->getLocStart());
      if (FullLocation.isValid())
        llvm::outs() << "Found declaration at "
                     << FullLocation.getSpellingLineNumber() << ":"
                     << FullLocation.getSpellingColumnNumber() << "\n";
    }
    return true;
  }

  bool VisitFunctionDecl(FunctionDecl *func_decl) {
    std::string func_name = func_decl->getNameInfo().getName().getAsString();
    FullSourceLoc full_loc = Context->getFullLoc(func_decl->getLocStart());
    const FileEntry *entry = full_loc.getManager().getFileEntryForID(full_loc.getFileID());
    // only care about functions in this file (not in include files)
    if (entry != m_main_entry) return true;

    // now the functions we care about
    llvm::outs() << "Visiting function: " << func_name
                 << "at " << entry->getName() << " "
                 << full_loc.getSpellingLineNumber() << ":"
                 << full_loc.getSpellingColumnNumber() << "\n";

    FunctionVisitor visitor(Context);
    llvm::outs() << "Now a new visitor for the function.\n";
    visitor.TraverseDecl(func_decl);

    llvm::outs() << "Another way: manually go through AST\n";
    // Stmt *stmt = func_decl->getBody();
    
    return true;
  }

private:
  ASTContext *Context;
  const FileEntry *m_main_entry = nullptr;
};



class TokenConsumer : public clang::ASTConsumer {
public:
  explicit TokenConsumer(ASTContext *Context)
    : Visitor(Context) {}

  // this function is called when the whole translation unit is parsed
  virtual void HandleTranslationUnit(clang::ASTContext &Context) {
    // entry opint for the visitor
    // using dynamic dispatch
    // will automatically call WalkUpFromXXX(x) to recursively visit child nodes of x
    // returning false of TraverseXXX or WalkUpFromXXX will terminate the traversal
    llvm::outs() << "======\n" << "Visiting Translation Unit" << "\n";
    
    Visitor.TraverseDecl(Context.getTranslationUnitDecl());
  }
private:
  TokenVisitor Visitor;
};


class TokenAction : public clang::ASTFrontendAction {
public:
  virtual std::unique_ptr<clang::ASTConsumer>
  CreateASTConsumer(clang::CompilerInstance &Compiler, llvm::StringRef InFile) {
    return std::unique_ptr<clang::ASTConsumer>
      (new TokenConsumer(&Compiler.getASTContext()));
  }
};


int main(int argc, const char **argv) {

  CommonOptionsParser OptionsParser(argc, argv, MyToolCategory);
  
  ClangTool Tool(OptionsParser.getCompilations(), OptionsParser.getSourcePathList());
  llvm::outs() << "running tool" << "\n";
  // Tool.run(newFrontendActionFactory<clang::SyntaxOnlyAction>().get());
  Tool.run(newFrontendActionFactory<TokenAction>().get());
}
